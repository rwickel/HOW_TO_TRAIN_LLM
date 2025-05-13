from datasets import Dataset, load_dataset
import torch, os, multiprocessing
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
from trl import SFTTrainer, SFTConfig
import numpy as np

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)


set_seed(1234)
max_memory = {0: "18GB"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
    attn_imlementation = 'flash'
else:
    compute_dtype = torch.float16
    attn_imlementation = 'sdpa'

# Format the data to match the expected structure of openassistant-guanaco
def format_instruction(sample):
    text = ("<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>"
            "You are a helpful, respectful, and honest assistant.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>"
            f"{sample['question']}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>"
            f"{sample['answer']}<|eot_id|>"
            "<|end_of_text|>")   
    
    print(text)
    return text

def main():
    # --- Load model and tokenizer ---
    model_name = "Qwen/Qwen3-4B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|end_of_text|>"

    tokenizer.padding_side = "left"

    # --- Load your custom dataset ---
    custom_data = [
        {
            "question": "What is the full title of ISO 21448:2022?",
            "answer": "ISO 21448:2022(en) — Road vehicles — Safety of the intended functionality."
        },
        {
            "question": "What is the increasing trend that makes the safety of road vehicles more important, according to the introduction of ISO 21448?",
            "answer": "The increasing commonality of automated driving functions makes the safety of road vehicles increasingly important."
        },
        {
            "question": "What are some of the key components that autonomous vehicles rely on to build situational awareness and make decisions?",
            "answer": "These systems depend on sensors, complex algorithms, and electronic components."
        },
        {
            "question": "What is the primary focus of ISO 26262 concerning safety risks in vehicles?",
            "answer": "ISO 26262 addresses safety risks caused by system faults."
        },
        {
            "question": "What type of hazards does ISO 21448 primarily focus on in autonomous vehicles?",
            "answer": "ISO 21448 focuses on hazards that arise even when systems work as intended but have limitations in performance or design."
        },
        {
            "question": "What is the main goal of the Safety of the Intended Functionality (SOTIF) as defined in ISO 21448?",
            "answer": "The Safety of the Intended Functionality (SOTIF) aims to reduce unreasonable risks caused by such insufficiencies, including incorrect perception, lack of robustness, or unexpected algorithm behavior."
        },
        {
            "question": "What are some of the key phases in a system's life cycle where SOTIF requires measures to be implemented?",
            "answer": "SOTIF requires measures across the system’s life cycle: from design adjustments to testing under diverse scenarios and monitoring during operation."
        },
        {
            "question": "Besides technical aspects, what human factors does ensuring SOTIF involve considering?",
            "answer": "Ensuring SOTIF also involves considering human factors, such as driver misunderstanding or inattention."
        },
        {
            "question": "How do interactions with external systems relate to SOTIF?",
            "answer": "SOTIF also considers interactions with external systems like infrastructure communications."
        },
        {
            "question": "In what way does ISO 21448 complement ISO 26262?",
            "answer": "ISO 21448 complements ISO 26262 by addressing hazards linked to functional limitations."
        },
        {
            "question": "According to ISO 21448:2022 Table 1, what is the source of hazards that the ISO 26262 series primarily addresses?",
            "answer": "For hazards originating within the system, faults in electrical and electronic (E/E) systems are addressed by the ISO 26262 series."
        },
        {
            "question": "What type of functional issues within a system are addressed by ISO 21448 according to Table 1?",
            "answer": "Functional insufficiencies—such as gaps in the system’s intended functionality or performance—are addressed by this document, ISO 21448."
        },
        {
            "question": "Which standard, besides ISO 21448, covers incorrect or inadequate Human-Machine Interface (HMI) design?",
            "answer": "Incorrect or inadequate Human-Machine Interface (HMI) design is also covered by the European Statement of Principles on Human-Machine Interface."
        },
        {
            "question": "What specific area of artificial intelligence within vehicle systems falls under the scope of ISO 21448 regarding functional insufficiencies?",
            "answer": "Functional insufficiencies in artificial intelligence-based algorithms are specifically within the scope of ISO 21448."
        },
        {
            "question": "Can you give an example of a hazard arising directly from system technologies, and which standard might address it?",
            "answer": "Hazards arising directly from system technologies, such as the risk of eye damage from a lidar beam, are handled by specific standards, for example IEC 60825."
        },
        {
            "question": "What external factor related to users or other road participants is covered by ISO 21448?",
            "answer": "Reasonably foreseeable misuse by users or other road participants is covered by ISO 21448."
        },
        {
            "question": "Which standard addresses security-related hazards in vehicles, such as cyberattacks?",
            "answer": "Security-related hazards, such as cyberattacks exploiting vehicle vulnerabilities, are addressed by ISO/SAE 21434."
        },
        {
            "question": "Name some standards, besides ISO 21448, that address hazards caused by active infrastructure or vehicle-to-vehicle communication?",
            "answer": "Hazards caused by active infrastructure, vehicle-to-vehicle communication, or external systems fall under the scope of ISO 21448 as well as standards like ISO 20077, ISO 26262, and IEC 61508."
        },
        {
            "question": "What are some examples of hazards arising from the vehicle’s surroundings that are addressed by ISO 21448?",
            "answer": "Hazards arising from the vehicle’s surroundings—such as other road users, passive infrastructure, weather conditions, or electromagnetic interference—are addressed both by ISO 21448."
        },
        {
            "question": "Besides ISO 21448, which other standard series addresses hazards arising from the vehicle's surroundings?",
            "answer": "The ISO 26262 series, in addition to other relevant standards like ISO 7637-2, ISO 7637-3, ISO 11452-2, ISO 11452-4, and ISO 10605, also addresses hazards arising from the vehicle’s surroundings."
        },
        {
            "question": "According to the scope of ISO 21448:2022, what is the definition of Safety of the Intended Functionality (SOTIF)?",
            "answer": "SOTIF is defined as the absence of unreasonable risk from hazards caused by functional insufficiencies—whether due to incomplete specifications at the vehicle level or limitations in the design or performance of electrical and electronic (E/E) systems."
        },
        {
            "question": "What types of measures does ISO 21448 outline to achieve and maintain SOTIF?",
            "answer": "It outlines design, verification, validation, and operational measures needed to achieve and maintain SOTIF."
        },
        {
            "question": "To what types of functionalities in road vehicles does ISO 21448 apply?",
            "answer": "It applies to functionalities requiring situational awareness based on complex sensors and algorithms, including emergency intervention systems and driving automation from Levels 1 to 5."
        },
        {
            "question": "What types of vehicles are within the scope of ISO 21448?",
            "answer": "It covers E/E systems in production road vehicles (except mopeds)."
        },
        {
            "question": "Does ISO 21448 address reasonably foreseeable misuse of vehicle functionalities?",
            "answer": "Yes, it also addresses reasonably foreseeable misuse."
        },
        {
            "question": "How does ISO 21448 consider the impact of remote operation or back-office communication on safety?",
            "answer": "It addresses the impact of remote operation or back-office communication when they could create safety risks."
        },
        {
            "question": "What type of faults in vehicles are explicitly excluded from the scope of ISO 21448?",
            "answer": "It excludes faults covered by ISO 26262."
        },
        {
            "question": "Are cybersecurity threats within the scope of ISO 21448?",
            "answer": "No, cybersecurity threats are excluded."
        },
        {
            "question": "Does ISO 21448 cover hazards directly caused by system technologies, such as lidar eye damage?",
            "answer": "No, hazards directly caused by system technologies (e.g., lidar eye damage) are excluded."
        },
        {
            "question": "Are risks like electric shock or fire generally covered by ISO 21448?",
            "answer": "Risks like electric shock or fire are excluded unless caused by the intended functionality."
        },
        {
            "question": "Does ISO 21448 apply to deliberate misuse of vehicle systems?",
            "answer": "No, it does not apply to deliberate misuse."
        },
        {
            "question": "Are established vehicle safety systems like stability control or airbags within the scope of ISO 21448?",
            "answer": "It does not apply to established systems already covered by proven safety measures, such as stability control or airbags."
        },
        {
            "question": "According to ISO 21448:2022, what are the two main categories of 'functional insufficiency'?",
            "answer": "Functional Insufficiency refers to either an insufficiency in specification or a performance insufficiency at the vehicle or system (E/E element) level."
        },
        {
            "question": "What can functional insufficiencies lead to in terms of vehicle behavior or safety measures?",
            "answer": "These insufficiencies can lead to hazardous behavior or an inability to prevent, detect, or mitigate reasonably foreseeable misuse."
        },
        {
            "question": "What is the primary aim of SOTIF activities concerning functional insufficiencies?",
            "answer": "SOTIF activities aim to identify and evaluate such insufficiencies and their effects."
        },
        {
            "question": "What are 'triggering conditions' in the context of functional insufficiencies?",
            "answer": "Functional insufficiencies are triggered by certain conditions, which may result in hazardous events or failures to mitigate misuse."
        },
        {
            "question": "Can you provide an example given in the text of how functional insufficiencies and triggering conditions can lead to harm in an advanced driver-assistance system?",
            "answer": "In a Level 2 highway driving assist system, a functional insufficiency in the sensor could cause an incorrect trajectory, while another insufficiency in the driver monitoring camera could fail to detect inattentiveness—leading to harm when both conditions are present with driver misuse."
        },
        {
            "question": "At what levels within a vehicle system can functional insufficiencies occur?",
            "answer": "Functional insufficiencies can occur at the vehicle level or element level."
        },
        {
            "question": "How can element-level functional insufficiencies contribute to safety issues?",
            "answer": "At the element level, they may produce output insufficiencies that contribute to hazardous behavior or inability to mitigate misuse, either alone or combined with other insufficiencies."
        },
        {
            "question": "Does ISO 21448 provide visual aids to understand the relationships between functional insufficiencies, triggering conditions, hazards, and harm?",
            "answer": "Yes, Figures in the standard illustrate these cause-and-effect relationships between functional insufficiencies, triggering conditions, hazards, and harm."
        }
    ]

    
    formatted_data = [{"text": format_instruction(item)} for item in custom_data]
    ds = Dataset.from_list(formatted_data)

    # --- Split into train and eval sets ---
    split_ds = ds.train_test_split(test_size=0.1)
    train_dataset = split_ds['train']
    eval_dataset = split_ds['test']

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=compute_dtype,
        max_memory=max_memory,
    )
    model.to(device)

    training_args = SFTConfig(
        output_dir="./Qwen3-4B_FTT",
        optim="adamw_torch",
        do_eval=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        per_gpu_eval_batch_size=2,
        eval_steps=10,
        log_level="debug",
        save_strategy="steps",
        logging_steps=25,
        learning_rate=1e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        num_train_epochs=100,
        warmup_ratio=0.01,
        lr_scheduler_type="linear",
        dataset_text_field="text",
        max_seq_length=512,
        save_safetensors=False, # save checkpoints as pytorch_model.bin
        resume_from_checkpoint=False # Set to False for a fresh run
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,             
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

if __name__ == "__main__":    
    
    main()