#!/bin/bash -l
#SBATCH --job-name=gpu-burn
#SBATCH --account=courses01-gpu
#SBATCH --reservation=UniAdelaideIntroToHPC-gpu
#SBATCH --partition=gpu
#SBATCH --time=00:07:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --output=gpu-burn-%j.out

# --- Load your Python/PyTorch/ROCm environment ---
module load pytorch/2.7.1-rocm6.3.3

echo "Visible GPUs to PyTorch:"
python - << 'EOF'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
EOF

# Run the burn script
    
srun -N 1 -n 1 -c 8 --gres=gpu:1 --gpus-per-task=1  python burn_gpu.py
