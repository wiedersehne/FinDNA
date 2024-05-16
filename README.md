# FinDNA
## Overview of FinDNA with self-supervised learning.
![image](https://github.com/wiedersehne/FinDNA/blob/main/findna.png)
Please load the pretained model "model_29_1000_4l_308_512_noiseandTL.pt" for all the downstream tasks.
## To evaluate FinDNA on GenomicBenchmarks, you need to:
1) Run ./data/genomic_benchmark.py to get train and test data for 8 tasks.
2) Run genomic_classification.py for fine tuning.

## To evaluate FinDNA on GUE, you need to:
1) Download dataset from https://github.com/MAGICS-LAB/DNABERT_2
2) Run gue_classification.py for fine tuning.

## To evaluate FinDNA on MTcDNA, you need to:
1) Run ./data/genome_process.py to create train and test dataset.
2) Run cdna_classification.py for fine tuning.
