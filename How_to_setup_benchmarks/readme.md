# GSM8K Evaluation Script

This Python script evaluates a language model on the [GSM8K dataset](https://huggingface.co/datasets/gsm8k) in two distinct modes:

- **Thinking Mode**: Encourages step-by-step reasoning by prompting the model to explain its process.
- **Non-Thinking Mode**: Directly prompts the model without reasoning scaffolding.

The script logs and saves all results, including correct and incorrect predictions.

---

## 🔍 About GSM8K

**GSM8K** (Grade School Math 8K) is a dataset consisting of **8,500 grade-school-level math word problems** designed to evaluate the mathematical reasoning capabilities of language models. It was introduced by OpenAI to test how well models can solve **multi-step arithmetic problems** and reasoning tasks.

- **8.5K examples**: 7,500 training samples and 1,000 test samples.
- **Natural language format**: Problems are written as they would appear on elementary school worksheets.
- **Solution format**: Each example includes a step-by-step reasoning breakdown, with the final answer clearly marked (e.g., `#### 12`).

### 🧠 Why GSM8K Matters:

- **Multi-step logic**: Models need to break down problems into smaller, logical steps to arrive at the correct solution.
- **Reasoning capability**: It challenges language models to **reason through arithmetic** and basic word problem understanding.
- **Performance evaluation**: Models that excel on GSM8K demonstrate strong **reasoning** and **generalization** abilities.

### 📘 Example Problem:

```json
{
  "question": "If there are 3 cars and each car has 4 tires, how many tires are there in total?",
  "answer": "Each car has 4 tires. So for 3 cars: 3 * 4 = 12. #### 12"
}```
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

