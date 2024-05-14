import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from Bio import SeqIO

bases = {"a": 0, "g":1, "c":2, "t":3, "n": 4, "<PAD>":5}

def merge():
    input_file_desert = "./data/MTcDNA/Chelonoidis_turtle"
    fasta_sequences = SeqIO.parse(open(input_file_desert), 'fasta')
    desert_sequences = []
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        desert_sequences.append(sequence)

    desert_df = pd.DataFrame(desert_sequences, columns=["sequence"])
    desert_df["length"] = desert_df['sequence'].apply(lambda x: len(x))
    desert_df["sequence"] = desert_df['sequence'].apply(lambda x: x.lower())
    desert_df = desert_df[desert_df["length"] > 5000]
    desert_df["class"] = 0

    input_file_island = "./data/MTcDNA/Gopherus_turtle"
    fasta_sequences_ff = SeqIO.parse(open(input_file_island), 'fasta')
    island_sequences = []
    for fasta in fasta_sequences_ff:
        name, sequence = fasta.id, str(fasta.seq)
        island_sequences.append(sequence)

    island_df = pd.DataFrame(island_sequences, columns=["sequence"])
    island_df["length"] = island_df['sequence'].apply(lambda x: len(x))
    island_df["sequence"] = island_df['sequence'].apply(lambda x: x.lower())
    island_df = island_df[island_df["length"] > 5000]
    island_df["class"] = 0

    input_file_Musculus = "./data/MTcDNA/Mus_musculus"
    fasta_sequences_ff = SeqIO.parse(open(input_file_Musculus), 'fasta')
    Musculus_sequences = []
    for fasta in fasta_sequences_ff:
        name, sequence = fasta.id, str(fasta.seq)
        Musculus_sequences.append(sequence)

    Musculus_df = pd.DataFrame(Musculus_sequences, columns=["sequence"])
    Musculus_df["length"] = Musculus_df['sequence'].apply(lambda x: len(x))
    Musculus_df["sequence"] = Musculus_df['sequence'].apply(lambda x: x.lower())
    Musculus_df = Musculus_df[Musculus_df["length"] > 5000]
    Musculus_df["class"] = 1

    input_file_Spretus = "./data/MTcDNA/Mus_spretus"
    fasta_sequences_ff = SeqIO.parse(open(input_file_Spretus), 'fasta')
    Spretus_sequences = []
    for fasta in fasta_sequences_ff:
        name, sequence = fasta.id, str(fasta.seq)
        Spretus_sequences.append(sequence)

    Spretus_df = pd.DataFrame(Spretus_sequences, columns=["sequence"])
    Spretus_df["length"] = Spretus_df['sequence'].apply(lambda x: len(x))
    Spretus_df["sequence"] = Spretus_df['sequence'].apply(lambda x: x.lower())
    Spretus_df = Spretus_df[Spretus_df["length"] > 5000]
    Spretus_df["class"] = 1

    df = pd.concat((desert_df, island_df, Musculus_df, Spretus_df), axis=0)
    print(df.head(5))
    print(df.shape)
    return df


def padding(sequence, desired_length):

    if len(sequence) > desired_length:
        return sequence[:desired_length]
    else:
        padding_character = 'N'

        # Pad the sequence
        padding_length = desired_length - len(sequence)
        padded_sequence = sequence + padding_character * padding_length
        return padded_sequence


def get_dna_csv(max_len):
    data_df = merge()
    X = [padding(s.upper(), max_len) for s in data_df.sequence.values]
    print(X[0])
    y = data_df["class"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    train_df = pd.DataFrame({"sequence": X_train, "label": y_train})
    test_df = pd.DataFrame({"sequence": X_test, "label": y_test})
    val_df = pd.DataFrame({"sequence": X_val, "label": y_val})

    train_df.to_csv(f"./data/MTcDNA/{max_len}/train.csv", index=None)
    val_df.to_csv(f"./data/MTcDNA/{max_len}/dev.csv", index=None)
    test_df.to_csv(f"./data/MTcDNA/{max_len}/test.csv", index=None)

get_dna_csv(8192)