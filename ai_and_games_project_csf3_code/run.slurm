#!/bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 2
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 4-00:00:00
#SBATCH --mem=240G
#SBATCH -J hex_train
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# --- 1. Environment Setup ---
echo "Setting up environment..."
module purge
module load libs/cuda
module load apps/binapps/conda/miniforge3

conda activate nlp_cw2_env

# --- 2. NVMe Storage Setup ---
echo "Setting up fast storage in $TMPDIR..."
WORK_DIR="$TMPDIR/hex_job_$SLURM_JOB_ID"
mkdir -p "$WORK_DIR"

echo "Copying files to $WORK_DIR..."
# Copy the training script
cp "$SLURM_SUBMIT_DIR/trainA100.py" "$WORK_DIR/"

# Copy the source code directory (src/)
# We need the whole folder structure for imports to work
cp -r "$SLURM_SUBMIT_DIR/src" "$WORK_DIR/"

# Copy existing model if it exists (to resume training)
if [ -f "$SLURM_SUBMIT_DIR/model_hpc.pt" ]; then
    echo "Found existing model, copying..."
    cp "$SLURM_SUBMIT_DIR/model_hpc.pt" "$WORK_DIR/"
fi

cd "$WORK_DIR"

# --- 3. Execution ---
echo "Starting training on $(hostname)"

# EXPORT THE SAVE DIRECTORY SO PYTHON KNOWS WHERE TO BACKUP
export SAVE_DIR="$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=24

# Run the training script
python -u trainA100.py

# --- 4. Save Results ---
echo "Training complete. Copying results back..."

# Copy the trained model back
if [ -f "model_hpc.pt" ]; then
    cp "model_hpc.pt" "$SLURM_SUBMIT_DIR/"
    echo "Model saved successfully."
else
    echo "Error: model_hpc.pt not found!"
fi

echo "Done."
