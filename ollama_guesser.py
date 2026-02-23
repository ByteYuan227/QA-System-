import logging
from collections import Counter, defaultdict

from parameters import Parameters
from guesser import Guesser
from gpr_guesser import kINSTRUCTIONS

import dspy

class OllamaGuesser(Guesser):
    # TODO(jbg): Make it so that this can cache like GPR guesser, but also include the name of the model
    def __init__(self):        
        self._model = None
    
    def set_ollama_model(self, model, instructions=kINSTRUCTIONS):
        self._model = model
        self._instructions = instructions

    def query(self, question):
        from ollama import chat
        from ollama import ChatResponse
        
        assert self._model is not None, "You need to set the model first"

        response: ChatResponse = chat(model='gemma3:4b', messages=[
            {
                'role': 'user',
                'content': self._instructions  + "\n\nQuestion %s" % question,
            },
        ])
        
        return response

def validate_answer(example: dspy.Example, pred, trace=None):
    from eval import rough_compare

    correct = rough_compare(example.answer, pred.best_guess)

    # Simplified bonus logic to focus on direct correctness
    if correct:
        return 1.0 + float(pred.confidence) ** 2
    else:
        return - (float(pred.confidence) ** 2)

   
class CalibratedRAG(dspy.Module):
    def __init__(self):
        # We replace the complex multi-step generation with a strict, single-step prediction
        # This prevents local models from failing JSON parsing and outputting markdown code fences
        class StrictPrediction(dspy.Signature):
            """
            Answer the quiz bowl question based on the retrieved context.
            CRITICAL INSTRUCTION: You MUST output ONLY valid data fields. 
            DO NOT use Markdown code blocks (e.g., ```json). 
            DO NOT include any conversational text, explanations, or formatting tags.
            """
            question: str = dspy.InputField()
            contexts: str = dspy.InputField(desc="Retrieved hints from TF-IDF")
            reasoning: str = dspy.OutputField(desc="Short evidence-based rationale")
            best_guess: str = dspy.OutputField(desc="Single best entity name ONLY")
            confidence: float = dspy.OutputField(desc="Probability that the guess is correct (0.0 to 1.0)")

        self.topk = None
        self.retrievers = {}
        # ChainOfThought automatically adds reasoning capability to the strict signature
        self.predictor = dspy.ChainOfThought(StrictPrediction)

    def forward(self, question, **kwargs):
        # 1. Handle cases where question is a list of sentences (from dspy.Example)
        if isinstance(question, list):
            question_text = " ".join(question)
        else:
            question_text = question

        # 2. Directly retrieve context using the stringified question
        contexts = ""
        for retriever_name in self.retrievers:
            # TfidfGuesser expects a string, not a list!
            results = self.retrievers[retriever_name](question_text, self._topk)
            for result in results:
                contexts += f"Hint from {retriever_name}: {result['guess']} (Confidence: {result['confidence']:.2f})\n"
            
        # 3. Make the prediction based on the retrieved contexts
        prediction = self.predictor(question=question_text, contexts=contexts)
        return prediction

    def init_retriever(self, name:str, model_filename:str, topk:int=1):
        """
        Add a new retriever to the system.

        Parameters:
        - name: The key of the retriever
        - filename: Where to load the retriever from (filename)
        - topk: How many retrieved results to get from the retriever per query
        """
        logging.info("Loading retriever %s as %s (top k=%i)" % (model_filename, name, topk))  
        from tfidf_guesser import TfidfGuesser 
        retriever = TfidfGuesser(model_filename)
        self.retrievers[name] = retriever 
        self.retrievers[name].load()
        if self.topk is None:    
            self._topk = topk    
        else:                    
            assert self._topk == topk 
    
class OllamaDspy(Guesser):
    def __init__(self, params: Parameters):
        self.filename = params.ollama_guesser_filename
        self._qa_metric = validate_answer
        self._confidence_metric = lambda x, y: (x - y) ** 2
        self._model = CalibratedRAG()
        self._optimized = None
        self.set_ollama_model(params.ollama_guesser_model)

        # Force using BootstrapFewShotWithRandomSearch or BootstrapFewShot for stability
        if params.ollama_guesser_opt_algorithm == 'MIPROv2':
            logging.warning("MIPROv2 is too slow/unstable for local models. Falling back to BootstrapFewShot.")
            from dspy.teleprompt import BootstrapFewShot
            self._teleprompter = BootstrapFewShot(
                metric=validate_answer,
                max_bootstrapped_demos=3,
                max_labeled_demos=3
            )
        else:
            from dspy.teleprompt import BootstrapFewShotWithRandomSearch
            self._teleprompter = BootstrapFewShotWithRandomSearch(
                metric=validate_answer,
                max_labeled_demos=10, # Reduced to prevent extreme memory usage and time
                num_threads=params.ollama_guesser_threads)
            
    @staticmethod
    def create_dataset(questions, answers):
        return [dspy.Example(question=x, answer=y).with_inputs("question") for x, y in zip(questions, answers)]

    def setup_retrieval(self, name:str, filename:str, topk:int):
        """
        Add a new retriever to the system.

        Parameters:
        - name: The key of the retriever
        - filename: Where to load the retriever from (filename)
        - topk: How many retrieved results to get from the retriever per query
        """
        self._model.init_retriever(name, filename, topk)
    
    def train(self, training_data, answer_field='page', split_by_sentence=True,
                    min_length=-1, max_length=-1, remove_missing_pages=True):
        Guesser.train(self, training_data, answer_field, split_by_sentence, min_length,
                      max_length, remove_missing_pages)

        # Slicing the dataset to a maximum of 50 examples to ensure fast compilation
        limit = min(len(self.questions), 50)
        dataset = self.create_dataset(self.questions[:limit], self.answers[:limit])
        
        self._optimized = self._teleprompter.compile(
            self._model,
            trainset=dataset,
        )

    def __call__(self, question, max_n_guesses=1):
        if self._optimized == None:
            model = self._model
        else:
            model = self._optimized

        # REMOVED: assert max_n_guesses == 1 
        # Evaluation scripts often request multiple guesses (n > 1) to calculate Recall@K.
        
        try:
            # Handle cases where question is a list of sentences (safety check)
            if isinstance(question, list):
                question_text = " ".join(question)
            else:
                question_text = question

            result = model(question=question_text)
            
            # Return a list containing our single best guess
            return [{"guess": result.best_guess, "confidence": float(result.confidence), "evidence": result.reasoning}]
        except Exception as e:
            logging.error(f"DSPy parsing failed: {e}")
            return [{"guess": "Unknown", "confidence": 0.0, "evidence": "Failed to parse LLM output"}]
            
    def set_ollama_model(self, model):
        self._lm = dspy.LM('ollama_chat/%s' % model, api_base='http://localhost:11434', api_key='')
        dspy.configure(lm=self._lm)

    def load(self):
        path = self.filename
        self._model.load("%s.json" % path)        

    def save(self):
        path = self.filename
        if self._optimized is None:
            self._model.save("%s.json" % path)
        else:
            self._optimized.save("%s.json" % path)
        
if __name__ == "__main__":
    import argparse
    import gzip
    import json
    
    from guesser import kTOY_DATA

    from dspy.evaluate import Evaluate

    from nltk.tokenize import sent_tokenize

    from parameters import load_questions, add_guesser_params, add_general_params, add_question_params, setup_logging, instantiate_guesser

    logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='Question Type')
    add_general_params(parser)
    guesser_params = add_guesser_params(parser)
    add_question_params(parser)

    flags = parser.parse_args()
    guesser_params["ollama"].load_command_line_params(flags)

    questions = load_questions(flags)
    od = instantiate_guesser("Ollama", flags, False)

    # First, let's show some examples
    for qq in kTOY_DATA["train"]:
        result =  od(qq["text"])
        print(qq, result)

    # TODO: can we use some of the more standard question flags for this?
    with gzip.open(flags.ollama_guesser_dev_data) as infile:
        questions = json.load(infile)
        
        dev = od.create_dataset([sent_tokenize(x["text"])[:-2] for x in questions[:flags.ollama_guesser_validation_ex]],
                                [x["page"] for x in questions[:flags.ollama_guesser_validation_ex]])

        evaluator = Evaluate(devset=dev, num_threads=flags.ollama_guesser_threads,
                             display_progress=True, display_table=10, provide_traceback=True)

        print("PRETRAIN: %f" % evaluator(od._model, metric=validate_answer))

    
    od.train(questions[flags.ollama_guesser_validation_ex:])
    od.save()

    print("POSTTRAIN: %f" % evaluator(od._optimized, metric=validate_answer))