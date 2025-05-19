from datasets import load_dataset
import torch, os, multiprocessing
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
from trl  import SFTTrainer, SFTConfig

set_seed(1234)
max_memory = {0: "20GB"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_bf16_supported():
    # os.system('pip install flash_attn')
    compute_dtype = torch.bfloat16
    attn_imlementation = 'flash'
else:
    compute_dtype = torch.float16
    attn_imlementation = 'sdpa'

def main():
    # --- Load model and tokenizer ---
    model_name = "Qwen/Qwen3-1.7B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|pad_id|>"

    tokenizer.padding_side = "left"

    # --- Load dataset ---
    ds = load_dataset("timdettmers/openassistant-guanaco")
    start_idx = 0
    end_idx = 512
    num_rows = len(ds['train'])
    ds = ds['train'].select(range(start_idx, min(end_idx, num_rows)))    

    def process(row):
        row["text"] = row["text"] + tokenizer.eos_token
        return row

    ds = ds.map(process,            
                load_from_cache_file=False,
                num_proc=multiprocessing.cpu_count(),
    )

    # 👇 split into train and eval sets
    split_ds = ds.train_test_split(test_size=0.1)
    train_dataset = split_ds['train']
    eval_dataset = split_ds['test']

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        #attn_imlementation = attn_imlementation,
        torch_dtype=compute_dtype,
        max_memory=max_memory,
    )
    model.to(device)

    training_args = SFTConfig(
        output_dir="./Qwen3-1.7B_FTT",
        optim="adamw_torch",
        do_eval=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        per_gpu_eval_batch_size=3,
        log_level="debug",
        save_strategy="steps",    
        logging_steps= 25,
        learning_rate=1e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        num_train_epochs=3,
        warmup_ratio=0.01,
        lr_scheduler_type="linear",
        dataset_text_field="text",
        max_seq_length=512,
        save_safetensors=False, # save checkpoints as pytorch_model.bin
        resume_from_checkpoint=True
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset = train_dataset,
        eval_dataset  = eval_dataset, 
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
