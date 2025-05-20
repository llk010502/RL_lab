import torch
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

# Check for device availability
if torch.cuda.is_available():
    device = "cuda"
    print(f"Using CUDA with {torch.cuda.device_count()} device(s)")
else:
    device = "cpu"
    print("Using CPU (no GPU acceleration available)")

# Load a small subset of the dataset for testing
dataset = load_dataset("trl-lib/tldr", split="train[:100]")

model_name = "Qwen/Qwen2-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto',
)

# PPO configuration
ppo_config = PPOConfig(
    batch_size=8,
    mini_batch_size=2,
    learning_rate=1e-5,
    ppo_epochs=2,
    gradient_accumulation_steps=4,
    max_grad_norm=0.5,
    optimize_cuda_cache=device == "cuda",
    target_kl=0.1,
    use_score_scaling=True,
    use_score_norm=True,
    only_optimize_generated=True,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
)

if __name__ == "__main__":
    print(f"Running PPO training iteration on {device}...")
    
    # Example of a training step
    for batch in ppo_trainer.dataloader:
        # Generate responses from a small batch
        query_tensors = batch["input_ids"][:2]
        
        # Generate responses
        response_tensors = ppo_trainer.generate(
            query_tensors,
            max_length=64,
            do_sample=True,
            top_k=10,
            temperature=0.7,
        )
        
        # Define a simple reward function
        rewards = [len(response) * 0.01 for response in response_tensors]
        
        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        print(f"PPO stats: {stats}")
        
        # Only run one iteration for this example
        break
    
    print(f"PPO test completed successfully on {device}!") 