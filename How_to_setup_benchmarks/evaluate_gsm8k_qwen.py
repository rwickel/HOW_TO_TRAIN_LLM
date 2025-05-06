"""
Author: Robert Wickel

Evaluates a language model on the GSM8K dataset in both "thinking" and "non-thinking" modes.
Logs and saves all outputs, including correct and incorrect predictions.
"""

import json
import torch
import re
import time
from transformers import AutoTokenizer, AutoModelForCausalLM


def evaluate_gsm8k(model, data_path="../data/gsm8k_test.jsonl", limit=None):
    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)

    with open(data_path, "r") as f:
        examples = [json.loads(line) for line in f]

    results = {
        "thinking_mode": {"correct": 0, "total": 0},
        "non_thinking_mode": {"correct": 0, "total": 0}
    }

    failed_cases = []
    all_cases = []

    def extract_answer_from_hash(text):
        match = re.search(r"####\s*(\S+)", text)
        return match.group(1).strip() if match else None

    def extract_answer(text):
        boxed_match = re.search(r"\\boxed\{(.*?)\}", text)
        if boxed_match:
            return boxed_match.group(1).strip()
        answer_match = re.search(r"The answer is[:\s]*([\d\.\-]+)", text)
        if answer_match:
            return answer_match.group(1).strip()
        numbers = re.findall(r"(\d+\.?\d*)", text)
        return numbers[-1].strip() if numbers else None

    def log_case(index, question, output_text, model_answer, expected, mode, duration):
        correct = model_answer == expected
        entry = {
            "index": index,
            "mode": mode,
            "duration_seconds": duration,
            "question": question,
            "model_output": output_text.strip(),
            "predicted": model_answer,
            "expected": expected,
            "correct": correct
        }
        all_cases.append(entry)

        if not correct:
            RED = "\033[91m"
            RESET = "\033[0m"
            print(f"{RED}[INCORRECT - {mode}] Q#{index} | Time: {duration:.2f}s{RESET}")
            print(f"{RED}Question: {question}{RESET}")
            print(f"{RED}Model Output: {output_text.strip()}{RESET}")
            print(f"{RED}Extracted Answer: {model_answer} | Expected: {expected}{RESET}\n")
            failed_cases.append(entry)

    for i, example in enumerate(examples[:limit]):
        question = example["question"]
        answer = example["answer"]
        expected = extract_answer_from_hash(answer)

        # -- Thinking mode --
        messages = [{
            "role": "user",
            "content": question + " Please reason step by step, and put your final answer within \\boxed{}."
        }]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32768, temperature=0.6, top_p=0.95, top_k=20)
        duration = time.time() - start_time

        output_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        model_answer = extract_answer(output_text)
        results["thinking_mode"]["total"] += 1
        if model_answer == expected:
            results["thinking_mode"]["correct"] += 1

        log_case(i, question, output_text, model_answer, expected, "thinking", duration)

        # -- Non-thinking mode --
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=8192, temperature=0.7, top_p=0.8, top_k=20)
        duration = time.time() - start_time

        output_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        model_answer = extract_answer(output_text)
        results["non_thinking_mode"]["total"] += 1
        if model_answer == expected:
            results["non_thinking_mode"]["correct"] += 1

        log_case(i, question, output_text, model_answer, expected, "non-thinking", duration)

    # -- Final summary --
    print("Final results:")
    print(f"Thinking mode accuracy: {results['thinking_mode']['correct'] / results['thinking_mode']['total']:.4f}")
    print(f"Non-thinking mode accuracy: {results['non_thinking_mode']['correct'] / results['non_thinking_mode']['total']:.4f}")

    return all_cases, failed_cases


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    all_cases, failed_cases = evaluate_gsm8k(model, limit=5)

    with open("gsm8k_all_cases.json", "w") as f:
        json.dump(all_cases, f, indent=2)

    with open("gsm8k_failed_cases.json", "w") as f:
        json.dump(failed_cases, f, indent=2)
