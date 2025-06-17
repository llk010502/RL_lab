# Search-R1 Fine-Tuning Guide (FinQA + LoRA + Per-Sample Retrieval)

This guide walks you through fine-tuning a language model on the [FinQA Combined dataset](https://huggingface.co/datasets/llk010502/FinQA_Combined_dataset) using LoRA and per-sample retrieval with the Search-R1 framework.

## Step 1: Environment Setup

1. **Install dependencies** (from the project root):
   ```bash
   pip install -r requirements.txt
   pip install datasets sentence-transformers faiss-cpu pyarrow
   ```

2. **(Optional) Set up CUDA and WandB** if using GPU and experiment tracking.

## Step 2: Prepare the Data

1. **Download and preprocess the FinQA dataset:**
   ```bash
   python scripts/prepare_finqa_data.py
   ```
   This will create `data/finqa_combined/train.parquet`, `val.parquet`, and `test.parquet` with the required columns: `question`, `answer`, and `information`.

## Step 3: Configure the Training Script

1. **Edit `train_ppo_fin.sh`** if needed:
   - Make sure `DATA_DIR` is set to `data/finqa_combined`.
   - Adjust batch sizes and LoRA parameters as needed for your hardware.

## Step 4: Run Fine-Tuning

1. **Start training:**
   ```bash
   bash train_ppo_fin.sh
   ```
   - Training will use LoRA adapters and per-sample retrieval (each sample's `information` field is indexed for local search).

## Step 5: Monitor and Evaluate

- Check logs and WandB (if enabled) for training progress.
- Validation is performed automatically during training.

---

## Notes
- The pipeline expects each sample to have `question`, `answer`, and `information` fields.
- Retrieval during training uses the `information` field for each sample (no external search engine needed).
- For custom datasets, follow the same format and preprocessing steps.


```bibtex
@article{jin2025search,
  title={Search-r1: Training llms to reason and leverage search engines with reinforcement learning},
  author={Jin, Bowen and Zeng, Hansi and Yue, Zhenrui and Yoon, Jinsung and Arik, Sercan and Wang, Dong and Zamani, Hamed and Han, Jiawei},
  journal={arXiv preprint arXiv:2503.09516},
  year={2025}
}
```

```bibtex
@article{jin2025empirical,
  title={An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents},
  author={Jin, Bowen and Yoon, Jinsung and Kargupta, Priyanka and Arik, Sercan O and Han, Jiawei},
  journal={arXiv preprint arXiv:2505.15117},
  year={2025}
}
```
