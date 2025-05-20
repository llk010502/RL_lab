import torch
from datasets import load_dataset
from trl import GRPOTrainer
from transformers import AutoTokenizer

# Check for device availability (CUDA or MPS)
if torch.cuda.is_available():
    device = "cuda"
    print(f"Using CUDA with {torch.cuda.device_count()} device(s)")
else:
    device = "cpu"
    print("Using CPU (no GPU acceleration available)")

# Load a small subset of the dataset for testing
dataset = load_dataset("trl-lib/tldr", split="train[:100]")

# Reward function
def reward_num_unique_chars(completions, **kwargs):
    return [len(set(c)) for c in completions]

model_name = "Qwen/Qwen2-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Initialize trainer with device-appropriate settings
trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=reward_num_unique_chars,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args={
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "gradient_checkpointing": True,
        "max_length": 128,
        "num_train_epochs": 1,
        "logging_steps": 10,
        "fp16": True,
        "output_dir": f"./output-grpo-{device}",
    },
)

# Train with minimal steps for testing
if __name__ == "__main__":
    print(f"Starting GRPO training on {device}...")
    trainer.train()
    print("Training complete!") 