import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import DataCollatorForLanguageModeling,TrainerCallback, TrainerState, TrainerControl
import gc
IGNORE_TOKEN_ID = -100  # same as LabelSmoother.ignore_index
max_memory = {0: "18GB"}


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# === Format the data to match Qwen format ===
# def format_instruction(sample):
#     return {
#         "input_text": (
#             "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
#             "You are a helpful, respectful, and honest assistant.<|eot_id|>"
#             "<|start_header_id|>user<|end_header_id|>"
#             f"{sample['question']}<|eot_id|>"
#             "<|start_header_id|>assistant<|end_header_id|>"
#             f"{sample['answer']}<|eot_id|><|end_of_text|>"
#         )
#     }

def format_instruction(sample): #QWEN chat template 
    text = (
        "<|im_start|>system\n"
        "You are a helpful, respectful, and honest assistant.<|im_end|>\n"
        f"<|im_start|>user\n{sample['question']}<|im_end|>\n"
        f"<|im_start|>assistant\n{sample['answer']}<|im_end|>"
    )
    return {"input_text": text}

if __name__ == "__main__": 

    clear_memory() 

    # --- Load model and tokenizer ---
    model_name = "Qwen/Qwen3-4B"

    # === Load tokenizer and model ===
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        attn_imlementation = 'flash'
    else:
        compute_dtype = torch.float16
        attn_imlementation = 'sdpa'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=compute_dtype,  # or bfloat16
        max_memory=max_memory,
    ).to(device)   

    raw_data = [
            {"question":"What is ISO 21448:2022 Safety of the Intended Functionality (SOTIF), and how does it address safety risks in autonomous vehicles that ISO 26262 does not cover?", "answer": "ISO 21448:2022(en) — Road vehicles — Safety of the intended functionality. Introduction. The safety of road vehicles is increasingly important as automated driving functions become more common. These systems depend on sensors, complex algorithms, and electronic components to build situational awareness and make decisions. While ISO 26262 addresses safety risks caused by system faults, ISO 21448 focuses on hazards that arise even when systems work as intended but have limitations in performance or design. The Safety of the Intended Functionality (SOTIF) aims to reduce unreasonable risks caused by such insufficiencies, including incorrect perception, lack of robustness, or unexpected algorithm behavior. SOTIF requires measures across the system’s life cycle: from design adjustments to testing under diverse scenarios and monitoring during operation. Ensuring SOTIF also involves considering human factors, such as driver misunderstanding or inattention, and interactions with external systems like infrastructure communications. ISO 21448 complements ISO 26262 by addressing hazards linked to functional limitations, helping ensure safer deployment of advanced vehicle systems."},
            {"question":"According to ISO 21448:2022 Table 1, how are safety-relevant hazards categorized based on their source? Explain which standards (e.g., ISO 26262, ISO 21448, IEC 60825) apply to each category, providing examples.", "answer": "ISO 21448:2022(en) Table 1 in the standard provides an overview of safety-relevant topics and the corresponding standards that address them, categorized by the source of hazard. For hazards originating within the system, faults in electrical and electronic (E/E) systems are addressed by the ISO 26262 series. Functional insufficiencies—such as gaps in the system’s intended functionality or performance—are addressed by this document, ISO 21448. Similarly, incorrect or inadequate Human-Machine Interface (HMI) design, which may lead to user confusion, overload, or inattentiveness, falls within the scope of ISO 21448 and is also covered by the European Statement of Principles on Human-Machine Interface. Functional insufficiencies in artificial intelligence-based algorithms are specifically within the scope of ISO 21448. Hazards arising directly from system technologies, such as the risk of eye damage from a lidar beam, are handled by specific standards, for example IEC 60825. For external factors, reasonably foreseeable misuse by users or other road participants is covered by ISO 21448 and, in some cases, by the ISO 26262 series. Security-related hazards, such as cyberattacks exploiting vehicle vulnerabilities, are addressed by ISO/SAE 21434. Hazards caused by active infrastructure, vehicle-to-vehicle communication, or external systems fall under the scope of ISO 21448 as well as standards like ISO 20077, ISO 26262, and IEC 61508. Finally, hazards arising from the vehicle’s surroundings—such as other road users, passive infrastructure, weather conditions, or electromagnetic interference—are addressed both by ISO 21448 and the ISO 26262 series, in addition to other relevant standards like ISO 7637-2, ISO 7637-3, ISO 11452-2, ISO 11452-4, and ISO 10605."},
            {"question":"What is the scope of ISO 21448:2022 regarding Safety of the Intended Functionality (SOTIF) for road vehicles, including what it covers and explicitly excludes?","answer": "ISO 21448:2022(en) — Road vehicles — Safety of the intended functionality. 1. Scope \nThis document provides a framework and guidance to ensure the Safety of the Intended Functionality (SOTIF), defined as the absence of unreasonable risk from hazards caused by functional insufficiencies—whether due to incomplete specifications at the vehicle level or limitations in the design or performance of electrical and electronic (E/E) systems. It outlines design, verification, validation, and operational measures needed to achieve and maintain SOTIF. It applies to functionalities requiring situational awareness based on complex sensors and algorithms, including emergency intervention systems and driving automation from Levels 1 to 5, covering E/E systems in production road vehicles (except mopeds). It also addresses reasonably foreseeable misuse and the impact of remote operation or back-office communication when they could create safety risks. The document excludes faults covered by ISO 26262, cybersecurity threats, hazards directly caused by system technologies (e.g., lidar eye damage), and risks like electric shock or fire unless caused by the intended functionality. It does not apply to deliberate misuse or established systems already covered by proven safety measures, such as stability control or airbags."},
            {"question":"According to ISO 21448:2022, how is 'functional insufficiency' defined?","answer": "Functional Insufficiency refers to either an insufficiency in specification or a performance insufficiency at the vehicle or system (E/E element) level. These insufficiencies can lead to hazardous behavior or an inability to prevent, detect, or mitigate reasonably foreseeable misuse. SOTIF activities aim to identify and evaluate such insufficiencies and their effects. Functional insufficiencies are triggered by certain conditions, which may result in hazardous events or failures to mitigate misuse. For example, in a Level 2 highway driving assist system, a functional insufficiency in the sensor could cause an incorrect trajectory, while another insufficiency in the driver monitoring camera could fail to detect inattentiveness—leading to harm when both conditions are present with driver misuse. Functional insufficiencies can occur at the vehicle level or element level. At the element level, they may produce output insufficiencies that contribute to hazardous behavior or inability to mitigate misuse, either alone or combined with other insufficiencies. Figures in the standard illustrate these cause-and-effect relationships between functional insufficiencies, triggering conditions, hazards, and harm."},
        ]    

    # Format and tokenize
    dataset = Dataset.from_list(raw_data).map(format_instruction)

    def tokenize(example):
        enc = tokenizer(
            example["input_text"],
            return_tensors="pt",
            padding="max_length",
            max_length=512,
            truncation=True
        )
        enc["labels"] = enc["input_ids"].clone()
        enc["labels"][enc["attention_mask"] == 0] = IGNORE_TOKEN_ID
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": enc["labels"][0],
        }

    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    # === Training Arguments ===
    training_args = TrainingArguments(
        output_dir="./qwen-sft_simple",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        logging_steps=1,
        save_strategy="steps",
        logging_strategy="steps",
        save_steps=100,
        num_train_epochs=8,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        eval_strategy="no",
        report_to="none",
        disable_tqdm=False,    
        save_total_limit=1,
        lr_scheduler_type="cosine",
        learning_rate=3e-5,   
        warmup_ratio=0.1,         
        
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # For causal LM like Qwen/LLaMA/GPT, set to False
    )

    # === Trainer ===
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,         
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024, 3)
    print(f"GPU Memory: {start_gpu_memory} GB, GPU Name: {gpu_stats.name}, GPU Capability: {gpu_stats.major}.{gpu_stats.minor}")

    print("Starting training...")
    trainer.train()

    end_gpu_memory = round(torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024, 3)
    print(f"Max GPU Memory Reserved During Training: {end_gpu_memory} GB")

    print("Saving best model...")
    trainer.save_model("./best_model")
    tokenizer.save_pretrained("./best_model", safe_serialization=True)
