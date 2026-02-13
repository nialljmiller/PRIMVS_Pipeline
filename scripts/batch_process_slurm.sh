#!/bin/bash
#SBATCH --job-name=primvs_reprocess
#SBATCH --account=galacticbulge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/primvs_%A_%a.out
#SBATCH --error=logs/primvs_%A_%a.err
#SBATCH --array=0-99  # Adjust based on total sources and chunk size

# Load necessary modules (adjust for Medbow environment)
module load python/3.10

# Activate virtual environment
source ~/python_projects/venv/bin/activate

# Set environment variables
export VIRAC_ROOT="/project/galacticbulge/PRIMVS"
export PRIMVS_OUTPUT="/project/galacticbulge/PRIMVS/reprocessed_catalog"
export PRIMVS_MODELS="/project/galacticbulge/PRIMVS/models"

# Define parameters
CATALOG_FITS="/project/galacticbulge/PRIMVS/catalog/PRIMVS_P.fits"
CHUNK_SIZE=10000  # Number of sources per job array task
START_IDX=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))

# Run the batch processing script
python process_catalog_chunk.py \
    --fits "$CATALOG_FITS" \
    --start "$START_IDX" \
    --count "$CHUNK_SIZE" \
    --output "primvs_chunk_${SLURM_ARRAY_TASK_ID}" \
    --n-processes "$SLURM_CPUS_PER_TASK"
