# Adaptive Delta & Standard (Ensemble) Watermarking for LLMs

This repository implements a comprehensive framework for generating, detecting, and attacking watermarked text from Large Language Models (LLMs). It unifies two complementary approaches:

1. **Standard Ensemble Watermarking** – A production-grade adaptation of SENSO, ACRO, RedGreen, and combined ensemble methods.
2. **Adaptive Delta Watermarking** – A specialized pipeline implementing the Adaptive Delta strategy for dynamic watermark adjustment.

---

## Attribution & Adaptation Note

The **Standard Watermarking Pipeline** (`run_all.py`, `modules/`, `pipeline_scripts/`) is adapted from the original _Ensemble Watermark Repository_ [(Link)](https://github.com/CommodoreEU/ensemble-watermark.git).

- **Original Source**: Core algorithmic logic in `modules/` and the early notebook workflow came from the original repository.
- **Enhancements in This Repo**:
  - Fully modularized pipeline
  - Single-command execution
  - Batch prompt generation
  - Automated attacks
  - Comparative visualization tooling

The **Adaptive Delta Pipeline** (`run_all_adaptive.py`, `adaptive_modules/`, `adaptive_pipeline/`) contains the self-synchronizing Adaptive Delta implementation.

---

## Project Architecture

Both pipelines share a unified repository structure. All data and norm files remain in the root directory and all model files must be stored inside the `models/` folder.

```

├── STANDARD PIPELINE (Adapted & Refactored)
│   ├── run_all.py               # Entry point for Ensemble watermark experiments
│   ├── pipeline_scripts/        # Orchestration: Generate → Detect → Attack → Plot
│   └── modules/                 # SENSO, ACRO, RedGreen & ensemble algorithms
│
├── ADAPTIVE DELTA PIPELINE
│   ├── run_all_adaptive.py      # Entry for Adaptive Delta experiments
│   ├── adaptive_pipeline/       # Adaptive orchestration workflow
│   └── adaptive_modules/        # Adaptive Delta watermark logic
│
├── MODELS DIRECTORY
│   ├── models/                  # Store all LLMs here
│   │   ├── llama-3.1-8b-instruct/
│   │   ├── mistral-7b-instruct/
        ├── llama-3.2-3b-instruct/
│   │   └── ...                  # Any additional models you use
│
├── ROOT-LEVEL RESOURCES (all data stored here)
│   ├── c4_prompts.json                     # Generated via create_prompt_dataset.py
│   ├── create_prompt_dataset.py            # Builds the prompt dataset
│   ├── Lancaster_sensorimotor_norms_for_39707_words.csv
│   ├── outputs/                            # Standard pipeline results
│   ├── self_sync_outputs*/                 # Adaptive Delta results
│   └── plots/                              # Visualizations from the notebooks

```

---

## Model Configuration

### Where to store your LLMs

All LLM weights **must be placed inside the `models/` directory**:

```

models/
├── llama-3.1-8b-instruct/
├── mistral-7b/
├── llama-3.2-3b/
└── ...

```

### Where to configure the model paths

Model paths are edited **inside the `CONFIG` block** of:

- `run_all.py` (for Standard / Ensemble watermarking)
- `run_all_adaptive.py` (for Adaptive Delta watermarking)

For example:

```python
CONFIG = {
    "MODEL_NAME": "models/llama-3.1-8b-instruct",
    "NUM_SAMPLES": 200,
    ...
}
```

---

## Key Features

### 1. Standard (Ensemble) Watermarking

- Supports **SENSO**, **ACRO**, **RedGreen**, and combined methods
- Configurable watermark **delta strengths**
- Batch experiment execution
- Automated detection and attack evaluation

### 2. Adaptive Delta Watermarking

- Dynamically adjusts watermark bias based on model confidence
- Dedicated adaptive modules and orchestration scripts

### 3. Robustness Testing & Analysis

- **Automated Attacks:**

  - Translation (EN → ES → EN)
  - T5 paraphrasing (span corruption)

- **Analysis Tools:**

  - P-value histograms
  - Perplexity–detection trade-off plots
  - Ensemble vs Adaptive comparisons

---

## Installation

### 1. Install Dependencies

```bash
pip install torch transformers pandas numpy matplotlib seaborn scipy statsmodels spacy bitsandbytes scikit-learn
```

Install `auto-gptq` separately if using quantized models.

### 2. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### 3. Required Root-Level Data Files

Ensure these exist at the **repository root**:

- `Lancaster_sensorimotor_norms_for_39707_words.csv`
- `updated_word_frequencies_with_percent.csv`
- `output_gen9`
- `c4_prompts.json` (generated via `create_prompt_dataset.py`)

---

## Usage

### Running the Standard Pipeline (Ensemble)

```bash
python run_all.py
```

Edit `CONFIG` inside `run_all.py` to change:

- model
- number of samples
- delta levels
- paraphrasing percentages

---

### Running the Adaptive Delta Pipeline

```bash
python run_all_adaptive.py
```

Edit `CONFIG` inside `run_all_adaptive.py` to configure:

- model
- adaptive thresholds
- synchronization settings

---

## Outputs & Artifacts

| Directory             | Description                                      |
| --------------------- | ------------------------------------------------ |
| `outputs/`            | Standard pipeline generation + detection results |
| `self_sync_outputs*/` | Adaptive Delta experiment logs + results         |
| `plots/`              | All generated visualizations                     |
