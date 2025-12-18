# #!/bin/bash
# # bash apps/main/debug_amd.sh

# PROJECT_FOLDER=$HOME

# cd $PROJECT_FOLDER/lingua
# source .venv/bin/activate

# CONFIG_FILE="apps/main/configs/toksuit_supertokenizer.yaml"

# # --------------------------------------------------------
# # AMD MI300X Configuration
# # --------------------------------------------------------
# export ROCM_HOME=/opt/rocm
# export HSA_OVERRIDE_GFX_VERSION=9.4.2

# # Make sure we're using the right GPU
# export CUDA_VISIBLE_DEVICES=0  # Single GPU
# export HIP_VISIBLE_DEVICES=0   # AMD equivalent

# # --------------------------------------------------------
# # Cache Directories
# # --------------------------------------------------------
# export TIKTOKEN_CACHE_DIR="$HOME/.cache/tiktoken"
# export HF_HOME="$HOME/.cache/huggingface"
# export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
# export HF_HUB_CACHE_DIR="$HOME/.cache/huggingface/hub"

# # export HF_HUB_OFFLINE=1
# # export TRANSFORMERS_OFFLINE=1

# # --------------------------------------------------------
# # RCCL (AMD's NCCL) Configuration - Critical for timeouts
# # --------------------------------------------------------
# export NCCL_TIMEOUT=3600           # 1 hour timeout
# export NCCL_BLOCKING_WAIT=1        # Synchronous operations
# export NCCL_ASYNC_ERROR_HANDLING=1 # Better error reporting

# # Network interface (important for rendezvous)
# export NCCL_SOCKET_IFNAME=lo       # Use loopback for single node
# export GLOO_SOCKET_IFNAME=lo       # Gloo backend also uses this

# # Debugging - reduce to WARN after it works
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

# # --------------------------------------------------------
# # wandb Configuration
# # --------------------------------------------------------
# export WANDB_MODE=offline 
# export WANDB_DIR="$HOME/wandb"
# wandb offline

# # --------------------------------------------------------
# # Distributed Training Parameters
# # --------------------------------------------------------
# MASTER_ADDR=127.0.0.1
# MASTER_PORT=29501
# RDZV_ID=101
# NPROC_PER_NODE=1
# NNODES=1

# echo "=========================================="
# echo "Training Configuration"
# echo "=========================================="
# echo "Master Address: $MASTER_ADDR"
# echo "Master Port: $MASTER_PORT"
# echo "Rendezvous ID: $RDZV_ID"
# echo "GPUs per node: $NPROC_PER_NODE"
# echo "Number of nodes: $NNODES"
# echo "Config file: $CONFIG_FILE"
# echo "=========================================="

# # Test GPU visibility
# echo "Testing GPU availability..."
# python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# # Test tiktoken cache
# echo "Testing tiktoken..."
# python -c "import tiktoken; tiktoken.encoding_for_model('gpt-4o')"

# echo "=========================================="
# echo "Starting training..."
# echo "=========================================="

# # Run with torchrun
# torchrun \
#     --nproc-per-node=$NPROC_PER_NODE \
#     --nnodes=$NNODES \
#     --node-rank=0 \
#     --rdzv_id=$RDZV_ID \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#     --max_restarts=0 \
#     --log_dir=$HOME/lingua/logs \
#     -m apps.main.train \
#     config=$CONFIG_FILE

#!/bin/bash

PROJECT_FOLDER=$HOME

cd $PROJECT_FOLDER/lingua
source .venv/bin/activate

CONFIG_FILE="apps/main/configs/debug_amd.yaml"

# --------------------------------------------------------
# AMD MI300X Configuration
# --------------------------------------------------------
export ROCM_HOME=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=9.4.2

export CUDA_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0

# --------------------------------------------------------
# Network Configuration - Force IPv4
# --------------------------------------------------------
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET

# Critical: Set the master address for c10d store
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501

# --------------------------------------------------------
# Cache Directories
# --------------------------------------------------------
export TIKTOKEN_CACHE_DIR="$HOME/.cache/tiktoken"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
export HF_HUB_CACHE_DIR="$HOME/.cache/huggingface/hub"

# --------------------------------------------------------
# RCCL Configuration
# --------------------------------------------------------
export NCCL_TIMEOUT=3600
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

# More aggressive socket binding
export NCCL_IB_DISABLE=1           # Disable InfiniBand for single node
export NCCL_P2P_DISABLE=0          # Enable P2P (even for single GPU)
export NCCL_SHM_DISABLE=0          # Enable shared memory

# Debugging
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# --------------------------------------------------------
# wandb Configuration
# --------------------------------------------------------
export WANDB_MODE=offline 
export WANDB_DIR="$HOME/wandb"
wandb offline

# --------------------------------------------------------
# Distributed Training Parameters
# --------------------------------------------------------
RDZV_ID=101
NPROC_PER_NODE=1
NNODES=1

echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Rendezvous ID: $RDZV_ID"
echo "GPUs per node: $NPROC_PER_NODE"
echo "Number of nodes: $NNODES"
echo "Config file: $CONFIG_FILE"
echo "=========================================="

# Test tiktoken
echo "Testing tiktoken..."
python -c "import tiktoken; tiktoken.encoding_for_model('gpt-4o')"

echo "=========================================="
echo "Starting training..."
echo "=========================================="

# Use static rdzv_endpoint with explicit IPv4
torchrun \
    --nproc-per-node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node-rank=0 \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    --rdzv_id=$RDZV_ID \
    --rdzv_backend=static \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --max_restarts=0 \
    --log_dir=$HOME/lingua/logs \
    -m apps.main.train \
    config=$CONFIG_FILE