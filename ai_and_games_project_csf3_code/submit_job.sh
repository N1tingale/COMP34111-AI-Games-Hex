#!/bin/bash

# Submit the job and capture the output
# Expected output: "Submitted batch job 123456"
OUTPUT=$(sbatch run.slurm)
echo "$OUTPUT"

# Extract the Job ID (4th word)
JOB_ID=$(echo "$OUTPUT" | awk '{print $4}')

if [[ -z "$JOB_ID" ]]; then
    echo "Error: Could not capture Job ID."
    exit 1
fi

# Define the expected log file name based on #SBATCH -J hex_train and -o %x_%j.out
LOG_FILE="hex_train_${JOB_ID}.out"

echo "Job submitted. Waiting for log file: $LOG_FILE to be created..."

# Loop until the file exists
while [ ! -f "$LOG_FILE" ]; do
    sleep 1
done

echo "Log file created! Tailing output (Press Ctrl+C to stop watching)..."
echo "-----------------------------------------------------------------"

# Tail the file
tail -f "$LOG_FILE"
