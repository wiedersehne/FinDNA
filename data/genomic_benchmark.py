from genomic_benchmarks.data_check import list_datasets
from genomic_benchmarks.data_check import info
import torch
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from genomic_benchmarks.dataset_getters.pytorch_datasets import get_dataset, HumanNontataPromoters, HumanEnhancersCohn, DemoHumanOrWorm, DemoMouseEnhancers, DemoCodingVsIntergenomicSeqs, DrosophilaEnhancersStark, HumanEnhancersEnsembl


def encode_sequence(d, ds, dt):
    """
    First get the datasets with fixed length, and one-hot encode each sequence.
    Then save them as torch tensors: X, y.
    """
    sequences = []
    labels = []
    for data in ds:
        gene_to_number = lb.transform(list(data[0]))
        sequences.append(gene_to_number)
        labels.append(data[1])
    X = torch.from_numpy(np.array(sequences)).to(torch.int8)
    y = torch.from_numpy(np.array(labels)).to(torch.float16)

    torch.save(X, f"./data/{d}_X_{dt}.pt")
    torch.save(y, f"./data/{d}_y_{dt}.pt")

def encode_sequence_varied(d, ds, dt, length):
    """
    First get the datasets with varied length, and one-hot encode each sequence.
    Then pad the sequence to maximum lengths. [mouse_enhancer:4776; Human_Enhancer_ensembl:573
    Human Regulatory: 802; Human_OCR:593]
    Lastly save them as torch tensors: X, y.
    """
    print(len(ds))
    sequences = []
    labels = []
    for data in ds:
        gene_to_number = lb.transform(list(data[0]))
        # print(gene_to_number)
        padded_seq = np.pad(gene_to_number, ((0, length-len(gene_to_number)), (0, 0)))
        # print(padded_seq)
        # print(len(gene_to_number))
        # print(len(padded_seq))
        sequences.append(padded_seq)
        labels.append(data[1])

    X = torch.from_numpy(np.array(sequences)).to(torch.int8)
    y = torch.from_numpy(np.array(labels)).to(torch.float16)

    torch.save(X, f"./data/{d}_X_{dt}.pt")
    torch.save(y, f"./data/{d}_y_{dt}.pt")

datasets = list_datasets()
lb = LabelBinarizer()
lb.fit(['A', 'T', 'C', 'G', 'N'])
varied_lengths = {
    "dummy_mouse_enhancers_ensembl":4776,
    "human_enhancers_ensembl": 573,
    "human_ensembl_regulatory": 802,
    "human_ocr_ensembl": 593
}
for d in datasets:
    print(info(d, version=0))
    train_dset = get_dataset(d, split='train', version=0)
    test_dset = get_dataset(d, split='test', version=0)
    if d in ["demo_coding_vs_intergenomic_seqs", "demo_human_or_worm", "human_enhancers_cohn", "human_nontata_promoters"]:
        encode_sequence(d, train_dset, "train")
        encode_sequence(d, test_dset, "test")
    elif d in ["dummy_mouse_enhancers_ensembl", "human_enhancers_ensembl", "human_ensembl_regulatory", "human_ocr_ensembl"]:
        max_length = varied_lengths[d]
        encode_sequence_varied(d, train_dset, "train", max_length)
        encode_sequence_varied(d, test_dset, "test", max_length)

