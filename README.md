# TRL Testing Environment

This repository contains test scripts for evaluating TRL (Transformer Reinforcement Learning) with PPO and GRPO algorithms.

## Environment Setup

### Python 3.10 Setup

```bash
# Create virtual environment
python3.10 -m venv rl_env

# Activate virtual environment
source rl_env/bin/activate  # Linux/macOS
.\rl_env\Scripts\activate   # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### GPU Environment (Optional)

For hardware acceleration with NVIDIA GPUs:

1. Install CUDA and cuDNN
2. Uncomment GPU packages in requirements.txt:
   - triton
   - deepspeed
   - flash-attn
3. Reinstall requirements: `pip install -r requirements.txt`

## Test Scripts

### PPO (Proximal Policy Optimization)

```bash
python test_ppo.py
```

### GRPO (Generative Reward Proximal Optimization)

```bash
python test_grpo.py
```

## Notes

- Scripts automatically detect and use GPU if available
- Both test scripts use Qwen/Qwen2-0.5B-Instruct model as an example
- Using TRL 0.17.0 which includes advanced trainers like GRPO, PPO, and more
- These scripts are for testing environment compatibility only 
- For GRPO to work correctly, you may need Hugging Face authentication 