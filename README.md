# Adaptive & Standard Watermarking for LLMs

This repository contains a comprehensive pipeline for generating, detecting, and analyzing watermarked text from Large Language Models (LLMs). It supports both **Standard Watermarking** (with various "delta" or strength levels) and **Self-Synchronizing (Adaptive) Watermarking**.

The project includes functionality for:
- **Watermarked Text Generation**: Integrating watermarks (SENSO, ACRO, RedGreen, and combinations) during text generation.
- **Detection**: Statistical detection of watermarks in generated text.
- **Robustness Testing**: Automated attacks using Translation and T5 Paraphrasing to test watermark persistence.
- **Analysis & Visualization**: Generating plots to analyze trade-offs between perplexity and detection, as well as robustness against attacks.

## Project Structure

- **`run_all.py`**: The main entry point for the **Standard Watermarking Pipeline**.
- **`run_all_adaptive.py`**: The main entry point for the **Self-Synchronizing Watermarking Pipeline**.
- **`pipeline_scripts/`**: Contains the scripts for the Standard Pipeline (generation, detection, attacks, plotting).
- **`adaptive_pipeline/`**: Contains the scripts for the Adaptive Pipeline.
- **`modules/` & `adaptive_modules/`**: Core logic for model loading, text generation, and watermarking algorithms.
- **`data/`**: Directory for input data (prompts).
- **`outputs/`**: Default output directory for the Standard Pipeline.
- **`self_sync_outputs*/`**: Output directories for the Adaptive Pipeline.
- **`plots/`**: Directory for storing generated plots.

## Requirements

The project requires Python 3.8+ and the following libraries:

- `torch`
- `transformers`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scipy`
- `statsmodels`
- `spacy`
- `bitsandbytes`
- `auto-gptq` (for loading quantization models)
- `scikit-learn`

### Installation

1.  **Install Python dependencies:**

    ```bash
    pip install torch transformers pandas numpy matplotlib seaborn scipy statsmodels spacy bitsandbytes auto-gptq scikit-learn
    ```

    *Note: You may need to install a specific version of `auto-gptq` depending on your CUDA version.*

2.  **Download Spacy model:**

    The project uses the `en_core_web_sm` model for linguistic processing.

    ```bash
    python -m spacy download en_core_web_sm
    ```

3.  **Data Files:**

    Ensure the following files are present in the root directory:
    - `Lancaster_sensorimotor_norms_for_39707_words.csv`: Sensorimotor norms data.
    - `c4_prompts.json`: Input prompts for text generation (referenced in config).

## Usage

### 1. Standard Watermarking Pipeline

This pipeline runs the standard watermarking experiments, including generation, detection, and robustness attacks.

To run the full pipeline:

```bash
python run_all.py
```

**Configuration:**
You can modify the `CONFIG` dictionary in `run_all.py` to change:
- `NUM_SAMPLES`: Number of text samples to generate (default is 5 for testing).
- `T5_REPLACE_PERCENTAGE`: Percentage of text to replace in T5 attacks.
- Output filenames and directories.

**Pipeline Steps:**
1.  **Setup & Load**: Loads the model and sensorimotor data.
2.  **Generate**: Generates text with various watermark configurations (SENSO, ACRO, etc.) and strengths (Weak, Medium, Strong).
3.  **Detect (Original)**: Runs detection on the generated text.
4.  **Attack**: Performs Translation and T5 Paraphrasing attacks.
5.  **Detect (Attacked)**: Runs detection on the attacked text.
6.  **Plot**: Generates visualization of results (boxplots, heatmaps, trade-off scatter plots).

### 2. Self-Synchronizing (Adaptive) Pipeline

This pipeline runs experiments for the self-synchronizing watermark, which adapts to the text structure.

To run the adaptive pipeline:

```bash
python run_all_adaptive.py
```

**Configuration:**
Modify the `CONFIG` dictionary in `run_all_adaptive.py` to change:
- `NUM_SAMPLES`: Number of samples.
- `SECRET_KEY_BITS`: Parameters for the hashing/key mechanism.
- `MODEL_NAME`: The specific model to use (e.g., Llama-3.2-3B-Instruct).

## Outputs

Results are saved in the configured output directories (default `outputs/` or `self_sync_outputs*/`).

Key output files include:
- `generation_results.json`: Generated text and metadata.
- `detection_results.json`: Detection scores for original text.
- `attacked_*.json`: Text after attacks.
- `detection_results_attack_*.json`: Detection scores after attacks.
- `*.png`: Various plots visualizing the performance.
