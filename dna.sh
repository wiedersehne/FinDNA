#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --account=share-ie-idi
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --mem=128000
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="contrast"
#SBATCH --output=mega.txt
#SBATCH --mail-user=tong.yu@ntnu.no
#SBATCH --mail-type=ALL


WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge
module load Anaconda3/2023.09-0
source activate monica
conda activate monica

# python3 -u enformer_veclf.py
# echo "python3"
# wandb agent tonyu/VE_20000/cvgt4a2q
# python3 -u pretraining_hg38.py --problem Pretraining --model Revolution
# python3 -u pretraining.py
#python3 pretraining_large.py￼
python3 genomic_classification.py
# python3 genomic_benchmark.py
# python3 hyper_search.py
# python3 cdna_classification.py
# wandb agent tonyu/VE_20000/o7qq8ejx#
# python3 genomic_classification.py
# wandb agent tonyu/Human_promoter/nsdo9xmd
# wandb agent tonyu/Human_cohn/5bu0evqi
# wandb agent tonyu/Human_worm/b0ixhb15
# wandb agent tonyu/Human_worm/a37jprb4
# python3 generate_pretrain_1m.py
# wandb agent tonyu/Human_cohn/x02jvqgd
# python contrastive_pretraining.py
# python contrastive_classification.py
#python pretraining.py