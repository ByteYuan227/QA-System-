from functools import partial
from collections import defaultdict
import logging

from typing import Dict, List

from buzzer import BuzzerParameters

from buzzer import Buzzer

import torch
from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer


class LoraBertParameters(BuzzerParameters):
    def __init__(self, customized_params=None):
        BuzzerParameters.__init__(self)
        self.name = "lorabert_buzzer"
        if customized_params:
            self.params += customized_params
        else:
            lorabert_params = [("rank", int, 16, "Rank of LoRA adaptation"),
                               ("base_model", str, "distilbert-base-uncased", "The HF model we will adapt"),
                               ("alpha", float, 1.0, "")]
            self.params += lorabert_params
                               

    def __setitem__(self, key, value):
        assert hasattr(self, key), "Missing %s, options: %s" % (key, dir(self))
        setattr(self, key, value)
           
    def set_defaults(self):
        for parameter, _, default, _ in self.params:
            name = "%s_%s" % (self.name, parameter)
            setattr(self, name, default)                


def initialize_base_model(helper_function=AutoModelForSequenceClassification,
                          model_name="distilbert-base-uncased"):
    """
    Initialize a BERT model and corresponding tokenizer.
    """
    model = helper_function.from_pretrained(model_name, num_labels=2)

    # Freeze the model parameters completely to save memory and time
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float):
        """
        Initialize a LoRA with two weight matrices whose product is the same as the original parameter matrix.
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha

        # Initialize A with random normal and B with zeros so initial delta is 0
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        torch.nn.init.normal_(self.A, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the linear layer's original result, then add the low-rank delta
        """
        assert x.shape[-1] == self.in_dim, "Input dimension %s does not match input dimension %i" % (str(x.shape), self.in_dim)

        # Compute the low-rank delta with alpha scaling
        delta = (x @ self.A @ self.B) * self.alpha
        return delta


class LinearLoRA(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear, rank: int, alpha: float):
        """
        Initialize a Linear layer with LoRA adaptation.
        """
        super().__init__()
        self.linear = linear

        # Initialize the LoRA layer
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adatpation.
        """
        result = self.linear(x)

        # Add the LoRA delta
        delta = self.lora(x)
        return result + delta


def add_lora(model: torch.nn.Module, rank: int, alpha: float, 
             modules_to_adapt: Dict[str, List[str]] = {"attention": ["q_lin", "k_lin", "v_lin", "out_lin"], "ffn": ["lin1", "lin2"]}):
    """
    Add LoRA layers to a PyTorch model dynamically.  
    """
    # Grab the layers list from either typical DistilBERT or the dummy BERT from tests
    if hasattr(model, "layer"):
        layers = model.layer
    else:
        layers = []

    # Inject LoRA
    for layer in layers:
        for component_name, module_names in modules_to_adapt.items():
            if hasattr(layer, component_name):
                component = getattr(layer, component_name)
                for module_name in module_names:
                    if hasattr(component, module_name):
                        original_linear = getattr(component, module_name)
                        if isinstance(original_linear, torch.nn.Linear):
                            lora_layer = LinearLoRA(original_linear, rank, alpha)
                            setattr(component, module_name, lora_layer)

    return model
                

class LoRABertBuzzer(Buzzer):
    def __init__(self, filename, run_length, num_guesses):
        super().__init__(filename, run_length, num_guesses)
        self._pipeline = None

    def initialize_model(self, model_name, rank, alpha):
        """
        Initialize the model and add LoRA layers.
        """
        self.model_name = model_name 
        self.rank = rank 
        self.alpha = alpha
        self.model, self.tokenizer = initialize_base_model(model_name=model_name)
        add_lora(self.model.distilbert.transformer, rank, alpha)

    def add_data(self, questions):
        None

    def build_features(self, history_length=0, history_depth=0):
        None
        
    def dataset_from_questions(self, questions, answer_field="page"):
        from eval import rough_compare
        from datasets import Dataset
        import pandas as pd
        
        metadata, answers, runs = self._clean_questions(questions, self.run_length, answer_field)

        all_guesses = {}
        logging.info("Building guesses from %s" % str(self._guessers.keys()))
        for guesser in self._guessers:
            all_guesses[guesser] = self._guessers[guesser].batch_guess(runs, self.num_guesses)
            logging.info("%10i guesses from %s" % (len(all_guesses[guesser]), guesser))
            assert len(all_guesses[guesser]) == len(runs), "Guesser %s wrong size" % guesser
            
        assert len(runs) == len(answers), "Runs (%i) don't match answers (%i)" % (len(questions), len(answers))
        assert len(metadata) == len(runs),  "Metadata (%i) don't match answers (%i)" % (len(metadata), len(answers))

        for guesser in all_guesses:
            assert len(all_guesses[guesser]) == len(runs), "Guesser %s length (%i) didn't match runs (%i)" % (guesser, len(all_guesses[guesser]), len(runs))
        
        dataset = []
        
        for guesser_id in all_guesses:
          for metadatum, answer, guesses, run in zip(metadata, answers, all_guesses[guesser_id], runs):
            example = {}

            for guess in guesses:
                if rough_compare(guess['guess'], answer):
                    correct = 1
                else:
                    correct = 0

                if answer is None:
                    answer = ""

                example["text"] = "%0.2f [SEP] %s [SEP] %s" % (guess['confidence'], guess['guess'], run)
                example["label"] = correct
                example["answer"] = answer
                example["id"] = metadatum["qanta_id"]
                example["length"] = len(run)
                example["guess"] = guess['guess']
                example["confidence"] = guess['confidence']

                dataset.append(example)

        dataframe = pd.DataFrame(data=dataset)
        print(dataframe.head())
        return Dataset.from_pandas(dataframe)

    def predict(self, questions):
        """
        Predict whether to buzz. Includes threshold tuning for higher Expected Wins.
        """
        if self._classifier is None:
            from transformers import pipeline
            device = 0 if torch.cuda.is_available() else -1
            # return_all_scores=True (or top_k=None in newer transformers) allows us to see the exact probability
            self._classifier = pipeline("text-classification", 
                                        model=self.model, 
                                        tokenizer=self.tokenizer, 
                                        device=device,
                                        top_k=None) 
            self._classifier.coef_ = [[]]

        inputs = self.dataset_from_questions(questions)
        text_inputs = [str(t) for t in inputs["text"]]
        
        # Get raw probabilities for all classes
        raw_predictions = self._classifier(text_inputs)
        
        predictions = []
        # TUNE THIS THRESHOLD: Lower means more aggressive buzzing
        BUZZ_THRESHOLD = 0.30

        
        for pred_list in raw_predictions:
            # Find the probability of LABEL_1 (which means "Buzz")
            buzz_prob = 0.0
            for output in pred_list:
                if output['label'] == 'LABEL_1' or output['label'] == '1':
                    buzz_prob = output['score']
                    break
            
            # Make the decision based on our custom threshold
            if buzz_prob > BUZZ_THRESHOLD:
                predictions.append(1)
            else:
                predictions.append(0)
        
        for i in range(len(questions)):
            questions[i]["run"] = inputs[i]["text"]
            questions[i]["guess"] = inputs[i]["guess"]
            questions[i]["answer"] = inputs[i]["answer"]
            questions[i].setdefault("id", i)
            
        return predictions, [], [], [pred == label for pred, label in zip(predictions, inputs["label"])], questions
    
    def train(self, finetune_dataset, eval_dataset):
        import numpy as np
        from transformers import DataCollatorWithPadding
        from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
        import evaluate
        from tqdm import tqdm

        accuracy = evaluate.load("accuracy")

        def preprocess_function(examples):
            return self.tokenizer(examples["text"], truncation=True)
        
        dataset = {}
        dataset["train"] = finetune_dataset.map(preprocess_function, batched=True)
        dataset["eval"] = eval_dataset.map(preprocess_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return accuracy.compute(predictions=predictions, references=labels)
                    
        training_args = TrainingArguments(  
            output_dir="models/lora_bert_buzzer",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            data_collator=data_collator,
            processing_class=self.tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()

    def save(self):
        # save only trained params (LoRA layers)
        model_state_dict = {
            name: param.detach().cpu() for name, param in self.model.named_parameters() if param.requires_grad
        }

        model_config = {
            "model_state_dict": model_state_dict,
            "model_name": getattr(self, "model_name", "distilbert-base-uncased"),
            "rank": getattr(self, "rank", None),
            "alpha": getattr(self, "alpha", None)
        }

        torch.save(model_config, "./models/lorabert.model")

    def load(self):
        # 1. Load the saved model file first to check the actual rank used during training
        saved_config = torch.load("./models/lorabert.model", map_location='cpu')
        
        # 2. Extract hyperparameters with fallback to CLI flags or defaults
        # This ensures the architecture matches the saved weights perfectly
        rank = saved_config.get("rank", getattr(self, "lorabert_buzzer_rank", 16))
        alpha = saved_config.get("alpha", getattr(self, "lorabert_buzzer_alpha", 1.0))
        base_model_name = saved_config.get("model_name", getattr(self, "lorabert_buzzer_base_model", "distilbert-base-uncased"))
        
        logging.info(f"Loading LoRA model with rank: {rank}, alpha: {alpha}")
        print(f"Success: Reconstructing model with Rank {rank} from saved checkpoint.")

        # 3. Initialize the base architecture using the extracted base_model_name
        self.model, self.tokenizer = initialize_base_model(model_name=base_model_name)
        
        # 4. Inject LoRA layers using the exact rank from the file (e.g., 32)
        add_lora(self.model.distilbert.transformer, rank, alpha)

        # 5. Load the trained weights into the reconstructed model
        self.model.load_state_dict(saved_config["model_state_dict"], strict=False)


if __name__ == "__main__":
    import gzip
    import argparse
    import json

    from parameters import add_buzzer_params, add_question_params, load_guesser, load_buzzer, load_questions, add_general_params, setup_logging, add_guesser_params
    
    parser = argparse.ArgumentParser()

    guesser_params = add_guesser_params(parser)
    buzzer_params = add_buzzer_params(parser)
    question_params = add_question_params(parser)

    add_general_params(parser)
    flags = parser.parse_args()    
    
    guesser = load_guesser(flags, guesser_params, load=True)    
    buzzer = load_buzzer(flags, buzzer_params)
    finetune_questions = load_questions(flags, secondary=False)
    dev_questions = load_questions(flags, secondary=True)
    
    setup_logging(flags)    

    # Train the model
    if finetune_questions:
      if flags.limit > 0:
        finetune_questions = finetune_questions[:flags.limit]
      finetune_dataset = buzzer.dataset_from_questions(finetune_questions)

    if dev_questions:
      if flags.limit > 0:
        dev_questions = dev_questions[:flags.limit]
      dev_dataset = buzzer.dataset_from_questions(dev_questions)
        
    buzzer.train(finetune_dataset, dev_dataset)
    buzzer.save()