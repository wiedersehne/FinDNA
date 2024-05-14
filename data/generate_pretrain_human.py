import torch
import numpy as np
from Bio import SeqIO
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from joblib import Parallel, delayed
import pysam
import multiprocessing
from functools import partial
from numpy.random import default_rng
rng = default_rng()


class UniformMasking():
    """ Pre-processing steps for pretraining revolution """
    def __init__(self, mask_prob):
        super().__init__()
        self.mask_ratio = mask_prob

    def __call__(self, instance):
        uniform_vec = np.random.uniform(size=len(instance))
        masked_vec = (uniform_vec <= self.mask_ratio).astype(int)

        uniform_vec2 = np.random.uniform(size=len(instance))
        random_vec = np.zeros(len(instance))
        same_vec = np.zeros(len(instance))
        random_vec[(masked_vec == 1) & (uniform_vec2 <= 0.1)] = 1
        same_vec[(masked_vec == 1) & (uniform_vec2 >= 0.9)] = 1
        real_vec = abs(masked_vec - random_vec - same_vec)
        random_vec = np.array(random_vec).astype(bool)
        real_vec = np.array(real_vec).astype(bool)

        instance[real_vec, :] = [0, 0, 0, 0, 0]
        instance[random_vec, :] = np.eye(5)[np.random.choice(5, sum(random_vec))]

        return instance, masked_vec

def generate_pairs(num):
    import pandas as pd
    import random

    # Read the CSV file containing chromosome lengths
    chromosome_lengths_df = pd.read_csv("./data/chromosomes.csv")

    chroms = []
    poses = []

    for _ in range(num):
        # Randomly select a chromosome
        random_chromosome = random.choice(chromosome_lengths_df["name"])

        # Get the length of the selected chromosome
        chromosome_length = chromosome_lengths_df.loc[
            chromosome_lengths_df["name"] == random_chromosome, "length"
        ].values[0]

        # Randomly select a position within the length range of the chromosome
        random_position = random.randint(1, chromosome_length)
        chroms.append(random_chromosome)
        poses.append(random_position)

    print(chroms[:10])
    print(poses[:10])
    return chroms, poses


def fetch_and_transform(position_and_chrom, length, lb, masking):
    """
        fetch one cunk of DNA sequences at given positions and chromosomes.
    """
    position, chrom = position_and_chrom
    genome = pysam.FastaFile('./data/hg38.fa')
    sequence = genome.fetch(chrom, position, position + length).lower()
    if len(sequence) == 0:
        print(f"Empty sequence at position {position}")
        return None
    # "transform sequence to one-hot encoding"
    gene_to_number = lb.transform(list(sequence)).astype("int8")
    # "Masking"
    masked_gene, mask = masking(np.array(gene_to_number))
    return masked_gene.astype("int8"), mask.astype("int8"), gene_to_number

def mask_chr_sequences(num, length, chroms, positions, split):
    """
        Parallelize the masking over 200k sequence.
    """
    masked_genes_train, genes_train, masks_train = [], [], []
    chunksize = int(num / multiprocessing.cpu_count()) # Or any other suitable value
    print(multiprocessing.cpu_count(), chunksize)
    with multiprocessing.Pool() as pool:
        results = pool.map(partial(fetch_and_transform, length=length, lb=lb, masking=masking), zip(positions, chroms), chunksize=chunksize)

    for masked_gene, mask, gene in results:
        if len(masked_gene) == length:
            masked_genes_train.append(masked_gene)
            masks_train.append(mask)
            genes_train.append(gene)

    X_train = torch.from_numpy(np.stack(masked_genes_train))
    M_train = torch.from_numpy(np.stack(masks_train))
    O_train = torch.from_numpy(np.stack(genes_train))

    print(X_train.shape, M_train.shape, O_train.shape)

    torch.save(X_train, f"./data/masked_{split}_{length}_10k.pt")
    torch.save(M_train, f"./data/mask_{split}_{length}_10k.pt")
    torch.save(O_train, f"./data/gene_{split}_{length}_10k.pt")



hg38_dict = SeqIO.to_dict(SeqIO.parse("./data/hg38.fa", "fasta"))
lb = LabelBinarizer()
lb.fit(['a', 't', 'c', 'g', 'n'])
chromosomes = [f"chr{i}" for i in range(1, 23)]
chromosomes.append("chrX")
chromosomes.append("chrY")
print(chromosomes)

masking = UniformMasking(0.3)
chroms, positions = generate_pairs(10000)
mask_chr_sequences(10000, 1000, chroms, positions, "valid")
