# QA System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-4.55%2B-yellow?logo=huggingface" alt="Transformers">
  <img src="https://img.shields.io/badge/DSPy-Enabled-green" alt="DSPy">
  <img src="https://img.shields.io/badge/Ollama-gemma3%3A4b-purple" alt="Ollama">
  <img src="https://img.shields.io/badge/LoRA-DistilBERT-red" alt="LoRA">
</p>


---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [File Structure](#-file-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
  - [Step 1: Train the TF-IDF Guesser](#step-1-train-the-tf-idf-guesser)
  - [Step 2: Optimize the Ollama DSPy Guesser](#step-2-optimize-the-ollama-dspy-guesser)
  - [Step 3: Train the LoRA-BERT Buzzer](#step-3-train-the-lora-bert-buzzer)
  - [Step 4: Evaluate](#step-4-evaluate)
- [Module Details](#-module-details)
  - [Guesser System](#guesser-system)
  - [Buzzer System](#buzzer-system)
- [Key Design Decisions](#-key-design-decisions)
- [Evaluation Metrics](#-evaluation-metrics)
- [Configuration & Parameters](#-configuration--parameters)

---

## 🔍 Project Overview

This project builds a **two-stage quiz bowl QA pipeline**:

1. **Guesser** — Given a (partial) trivia question, predict the most likely Wikipedia-page answer.
   - Stage 1: A **TF-IDF retriever** that performs fast lexical search over a training corpus.
   - Stage 2: An **LLM-powered RAG guesser** (`OllamaDspy`) using [DSPy](https://github.com/stanfordnlp/dspy) teleprompting to optimize prompts automatically on top of a local Ollama model (e.g., `gemma3:4b`).

2. **Buzzer** — Given the guesser's output at each partial reading of the question, decide whether to buzz (commit to the answer) or wait for more context.
   - Primary: A **LoRA fine-tuned DistilBERT** (`LoRABertBuzzer`) that frames buzzing as a sequence classification task.
   - Fallback: A lightweight **Threshold Buzzer** based on raw guesser confidence.

---

## 🏗 Architecture

```
Quiz Bowl Question (partial text)
         │
         ▼
┌─────────────────────────────────────┐
│          TF-IDF Guesser             │  ──► RAG hints (top-k candidates)
│  (sklearn, bigrams, cosine sim)     │
└─────────────────────────────────────┘
         │ hints
         ▼
┌─────────────────────────────────────┐
│    OllamaDspy (CalibratedRAG)       │  ──► answer + confidence
│  DSPy ChainOfThought + Teleprompter │
│  Ollama LLM (gemma3:4b / qwen3:4b) │
└─────────────────────────────────────┘
         │ (answer, confidence)
         ▼
┌─────────────────────────────────────┐
│       LoRABertBuzzer                │  ──► buzz decision
│  DistilBERT + LoRA (rank, alpha)    │
│  Hugging Face Trainer fine-tuning   │
└─────────────────────────────────────┘
         │
         ▼
      BUZZ / WAIT
```

---

## 📁 File Structure

```
prompt-ft/
├── guesser.py              # Base Guesser class, toy data, utilities
├── tfidf_guesser.py        # TF-IDF Guesser (sklearn, bigram, cosine similarity)
├── ollama_guesser.py       # DSPy-based Guesser using Ollama LLM + RAG
├── gpr_guesser.py          # GPR-based Guesser (alternative)
├── buzzer.py               # Base Buzzer class, run generation, featurization
├── lorabert_buzzer.py      # LoRA-BERT Buzzer (DistilBERT + custom LoRA layers)
├── logistic_buzzer.py      # Logistic Regression Buzzer (baseline)
├── threshold_buzzer.py     # Threshold-based Buzzer (simple confidence cutoff)
├── features.py             # Feature extractors (e.g., LengthFeature)
├── eval.py                 # Evaluation script (guesser precision/recall, buzzer expected wins)
├── parameters.py           # Centralized argparse config and object factory
├── requirements.txt        # Python dependencies
└── models/                 # Saved model artifacts
    ├── TfidfGuesser.*      # TF-IDF vectorizer, tfidf matrix, Q&A pickles
    ├── guesser.json        # Optimized DSPy prompt program
    └── lorabert.model      # LoRA fine-tuned BERT weights (PyTorch)
```

---

## ⚙️ Installation

### 1. Python Environment

```bash
./venv/bin/pip3 install -r requirements.txt
```

> Key dependencies: `torch`, `transformers>=4.55`, `dspy`, `ollama`, `scikit-learn`, `datasets`, `evaluate`, `accelerate>=0.26`

### 2. Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Pull the supported model:

```bash
ollama pull gemma3:4b
# Also available: qwen3:4b, llama3.2:3b
```

> **On HPC (no sudo):** Follow the [local Ollama install guide](https://medium.com/@abdulsalamelelu/executing-ollama-on-hpc-without-sudo-access-7eb6217e6fcb).

---

## 🚀 Quick Start

### Step 1: Train the TF-IDF Guesser

The TF-IDF guesser provides retrieval hints for the LLM and is the backbone of the RAG pipeline.

```bash
./venv/bin/python3 guesser.py \
  --guesser_type=Tfidf \
  --question_source=gzjson \
  --questions=../data/qanta.guesstrain.json.gz \
  --logging_file=guesser.log
```

Evaluate it:

```bash
./venv/bin/python3 eval.py \
  --evaluate guesser \
  --questions ../data/qanta.guessdev.json.gz \
  --cutoff 200 \
  --num_guesses 10 \
  --guesser_type Tfidf
```

### Step 2: Optimize the Ollama DSPy Guesser

Uses DSPy teleprompting to automatically optimize the prompt sent to the LLM.

```bash
# Quick test with 1000 examples
./venv/bin/python3 ollama_guesser.py --limit 1000

# Full training (remove --limit or increase)
./venv/bin/python3 ollama_guesser.py
```

The optimized prompts are saved to `models/guesser.json`.

Evaluate the DSPy guesser:

```bash
./venv/bin/python3 eval.py \
  --evaluate guesser \
  --questions ../data/qanta.buzztrain.json.gz \
  --limit 100 \
  --num_guesses 1 \
  --guesser_type Ollama
```

---

### Step 3: Train the LoRA-BERT Buzzer

Fine-tune DistilBERT with LoRA adapters to classify whether to buzz:

```bash
# With TF-IDF guesser (quick baseline)
./venv/bin/python3 lorabert_buzzer.py \
  --questions=../data/qanta.buzztrain.json.gz \
  --secondary_questions=../data/qanta.buzzdev.json.gz \
  --buzzer_guessers=Tfidf \
  --load=True \
  --buzzer_type='lorabert' \
  --limit=100

# With DSPy/Ollama guesser (recommended)
./venv/bin/python3 lorabert_buzzer.py \
  --questions=../data/qanta.buzztrain.json.gz \
  --secondary_questions=../data/qanta.buzzdev.json.gz \
  --buzzer_guessers=Ollama \
  --load=True \
  --buzzer_type='lorabert' \
  --limit=100
```

The fine-tuned model is saved to `models/lorabert.model`.

---

### Step 4: Evaluate

**Guesser evaluation:**

```bash
./venv/bin/python3 eval.py \
  --evaluate guesser \
  --questions ../data/qanta.guessdev.json.gz \
  --cutoff 200 --num_guesses 10 --guesser_type Tfidf
```

**Buzzer evaluation:**

```bash
./venv/bin/python3 eval.py \
  --evaluate buzzer \
  --questions ../data/qanta.buzzdev.json.gz \
  --limit 100 --num_guesses 1 \
  --buzzer_guessers=Tfidf \
  --buzzer_type='lorabert'
```

---

## 📦 Module Details

### Guesser System

| Module | Class | Description |
|--------|-------|-------------|
| `guesser.py` | `Guesser` | Abstract base: `train`, `batch_guess`, `save`, `load`, toy data |
| `tfidf_guesser.py` | `TfidfGuesser` | sklearn TF-IDF vectorizer with bigrams; cosine similarity retrieval |
| `ollama_guesser.py` | `OllamaDspy` | DSPy `CalibratedRAG` module: TF-IDF RAG → LLM → calibrated answer + confidence |
| `gpr_guesser.py` | `GprGuesser` | Gaussian Process Regression guesser (alternative backend) |

**`CalibratedRAG` (DSPy Module):**
- Uses `ChainOfThought` on a `StrictPrediction` signature
- Inputs: `question`, `contexts` (TF-IDF hits)
- Outputs: `reasoning`, `best_guess`, `confidence`
- Optimized via `BootstrapFewShot` teleprompter
- Scoring: `+1` for correct guess, `+(confidence²)` bonus, `-(confidence²)` penalty for wrong

---

### Buzzer System

| Module | Class | Description |
|--------|-------|-------------|
| `buzzer.py` | `Buzzer` | Base class: run generation, featurization, batch prediction |
| `lorabert_buzzer.py` | `LoRABertBuzzer` | DistilBERT + custom LoRA layers; Hugging Face `Trainer`-based fine-tuning |
| `threshold_buzzer.py` | `ThresholdBuzzer` | Confidence threshold + minimum length cutoff |
| `logistic_buzzer.py` | `LogisticBuzzer` | Logistic regression over hand-crafted features |

**LoRA Implementation:**

```
LoRALayer:   delta = (x @ A @ B) * alpha
LinearLoRA:  output = W(x) + delta
add_lora():  Replaces Linear layers in attention (q,k,v,out) and FFN (lin1,lin2)
```

The base model parameters are **frozen** (`requires_grad=False`); only the LoRA `A` and `B` matrices are trained.

**Buzzer Input Format:**

```
"{confidence:.2f} [SEP] {guess} [SEP] {question_run}"
```

---

## 🎯 Key Design Decisions

- **Bigram TF-IDF**: Enabled by default (`ngram_range=(1,2)`) for improved recall over training corpus.
- **DSPy Teleprompter**: Uses `BootstrapFewShot` for local model stability (MIPROv2 is too slow for local Ollama models).
- **LoRA rank=16, alpha=1.0**: Adapts all attention and FFN linear layers in each DistilBERT transformer block.
- **Buzz threshold = 0.30**: Lower than default (0.5) for more aggressive buzzing behavior.
- **Frozen base model**: All original BERT weights are frozen; only LoRA delta matrices are learned.
- **Answer normalization**: Strips articles ("the", "a", "an"), punctuation, and Unicode characters before comparison.

---

## 📊 Evaluation Metrics

### Guesser

| Metric | Description |
|--------|-------------|
| **Precision @1** | Was the top guess correct? |
| **Recall @k** | Was the correct answer in the top-k guesses? |
| **Calibration Error** | How well does confidence correlate with correctness? |

### Buzzer

| Metric | Description |
|--------|-------------|
| **Accuracy** | (best + waiting) / total |
| **Buzz Ratio** | (best - 0.5 × aggressive) / total |
| **Expected Wins** | Primary metric; computed via `1 + 0.0775t - 1.278t² + 0.588t³` (see [paper](https://arxiv.org/pdf/1904.04792)) |

**Buzzer outcome labels:**

| Label | Meaning |
|-------|---------|
| `best` | Correctly guessed AND buzzed |
| `timid` | Correctly guessed, did NOT buzz (missed points) |
| `aggressive` | Wrongly guessed AND buzzed (penalty!) |
| `waiting` | Wrongly guessed, correctly did NOT buzz |

---

## 🔧 Configuration & Parameters

All parameters are managed via `parameters.py`. Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--guesser_type` | `Tfidf` | `Tfidf`, `Ollama`, `Dan`, `gpr` |
| `--buzzer_type` | `logistic` | `lorabert`, `threshold`, `logistic` |
| `--buzzer_guessers` | `['Tfidf']` | Guessers feeding the buzzer |
| `--limit` | `-1` | Cap number of questions (useful for quick tests) |
| `--ollama_guesser_model` | `gemma3:4b` | Ollama model name |
| `--ollama_guesser_opt_algorithm` | `MIPROv2` | DSPy optimizer (`BootstrapFewShotWithRandomSearch` recommended) |
| `--lorabert_buzzer_rank` | `16` | LoRA rank |
| `--lorabert_buzzer_alpha` | `1.0` | LoRA scaling factor |
| `--threshold_buzzer_threshold` | `0.5` | Confidence threshold for `ThresholdBuzzer` |

---



