export DISABLE_VERSION_CHECK=1
# 4-GPU training
export CUDA_VISIBLE_DEVICES=0,1,2,3

# offline mode (use local checkpoints/datasets)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# export HF_ENDPOINT=https://hf-mirror.com
# set wandb project
export WANDB_PROJECT=MIMIGenRec-SFT
export WANDB_MODE=offline
export WANDB_API_KEY=""
set -x

# --- 0.5B ---
# llamafactory-cli train examples/train_full/Industrial_and_Scientific/industry_rec_full_sft_0.5b_dsz0.yaml
# llamafactory-cli train examples/train_full/Industrial_and_Scientific/industry_rec_full_sft_0.5b_dsz2.yaml
# llamafactory-cli train examples/train_full/Industrial_and_Scientific/industry_rec_full_sft_0.5b_dsz3.yaml

# --- 1.5B ---
# llamafactory-cli train examples/train_full/Industrial_and_Scientific/industry_rec_full_sft_1.5b_dsz0.yaml
# llamafactory-cli train examples/train_full/Industrial_and_Scientific/industry_rec_full_sft_1.5b_dsz2.yaml
# llamafactory-cli train examples/train_full/Industrial_and_Scientific/industry_rec_full_sft_1.5b_dsz3.yaml

# --- 3B ---
llamafactory-cli train examples/train_full/Industrial_and_Scientific/industry_rec_full_sft_3b_dsz0_4gpu_eq8.yaml
# llamafactory-cli train examples/train_full/Industrial_and_Scientific/industry_rec_full_sft_3b_dsz0.yaml
# llamafactory-cli train examples/train_full/Industrial_and_Scientific/industry_rec_full_sft_3b_dsz2.yaml
# llamafactory-cli train examples/train_full/Industrial_and_Scientific/industry_rec_full_sft_3b_dsz3.yaml
