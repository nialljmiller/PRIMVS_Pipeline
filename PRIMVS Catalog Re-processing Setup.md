# PRIMVS Catalog Re-processing Setup

This setup is designed to re-process the entire PRIMVS catalog on the Medbow supercomputer as efficiently as possible using SLURM job arrays.

## Prerequisites

1.  **Environment**: Ensure your virtual environment is set up and all dependencies are installed.
    ```bash
    cd /project/galacticbulge/PRIMVS/PRIMVS_Pipeline
    python3 -m venv venv
    source venv/bin/activate
    pip install -e .
    pip install -e "../StochiStats[full]"
    ```

2.  **Directories**: Ensure the output and logs directories exist.
    ```bash
    mkdir -p /project/galacticbulge/PRIMVS/reprocessed_catalog
    mkdir -p /project/galacticbulge/PRIMVS/PRIMVS_Pipeline/logs
    ```

## Processing Strategy

The re-processing is split into chunks using a **SLURM Job Array**. This allows for massive parallelism across multiple nodes on the supercomputer.

### 1. The Processing Script (`scripts/process_catalog_chunk.py`)
This script reads a specific slice of the `PRIMVS_P.fits` catalog and runs the pipeline on those sources.

### 2. The SLURM Batch Script (`scripts/batch_process_slurm.sh`)
This script defines the job parameters for Medbow. 
- **Array**: `--array=0-99` means 100 parallel jobs.
- **Chunk Size**: Each job processes 10,000 stars (adjust `CHUNK_SIZE` in the script based on the total catalog size).
- **Resources**: Each job uses 32 cores for internal multi-threading.

## Execution Steps

1.  **Configure Paths**: Open `scripts/batch_process_slurm.sh` and verify the paths for `VIRAC_ROOT`, `PRIMVS_OUTPUT`, and `CATALOG_FITS`.

2.  **Submit the Job**:
    ```bash
    sbatch scripts/batch_process_slurm.sh
    ```

3.  **Monitor Progress**:
    ```bash
    squeue -u $USER
    tail -f logs/primvs_<jobid>_<taskid>.out
    ```

4.  **Aggregate Results**:
    Once all array tasks are complete, merge the chunks into a single final catalog:
    ```bash
    python scripts/aggregate_chunks.py \
        --input-dir /project/galacticbulge/PRIMVS/reprocessed_catalog \
        --output /project/galacticbulge/PRIMVS/reprocessed_catalog/PRIMVS_REPROCESSED \
        --format fits
    ```

## Optimization Notes

- **Parallelism**: We use both job-level parallelism (SLURM Array) and node-level parallelism (Python `multiprocessing`).
- **Memory**: 64GB per node is requested, which should be plenty for 32 cores processing CSV/FITS data.
- **Time**: 24 hours is a safe limit for 10,000 stars per job, though it will likely finish much faster.
