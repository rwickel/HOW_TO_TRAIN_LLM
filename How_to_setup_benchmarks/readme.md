# GSM8K Evaluation Script

This Python script evaluates a language model on the [GSM8K dataset](https://huggingface.co/datasets/gsm8k) in two distinct modes:

- **Thinking Mode**: Encourages step-by-step reasoning by prompting the model to explain its process.
- **Non-Thinking Mode**: Directly prompts the model without reasoning scaffolding.

The script logs and saves all results, including correct and incorrect predictions.

---

##  Files

- `evaluate_gsm8k.py`: Main script to evaluate the model.
- `gsm8k_all_cases.json`: Saved results from both modes.
- `gsm8k_failed_cases.json`: Subset of examples where the model's answer was incorrect.

---

##  Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- A GPU with sufficient VRAM for large models (e.g., 16GB+)

Install dependencies:
```bash
pip install torch transformers
```
---
##  Run
    python evaluate_gsm8k_qwen.py --limit 50

