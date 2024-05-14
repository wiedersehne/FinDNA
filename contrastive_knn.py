from sklearn.neighbors import KNeighborsClassifier
from models.pretraining_model import Model4TSNE
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed(42)


def cls_augment(masked_gene, local_cls_number):
    N, L, D = masked_gene.shape
    cls_masked = torch.zeros(N, local_cls_number, D)

    masked_gene = torch.cat((masked_gene, cls_masked), 1)
    return masked_gene

class CustomDataset(torch.utils.data.Dataset):

  def __init__(self, X):
    super().__init__()

    self.X = X

  def __getitem__(self, idx):
    return self.X[idx] # In case you stored your data on a list called instances

  def __len__(self):
    return len(self.X)
  
task_list = ['demo_coding_vs_intergenomic_seqs', 'demo_human_or_worm', 'dummy_mouse_enhancers_ensembl', 'human_enhancers_cohn', 'human_enhancers_ensembl', 'human_ensembl_regulatory',  'human_nontata_promoters', 'human_ocr_ensembl']
  

if __name__ == "__main__":
    task = "human_nontata_promoters"
    pretrained_path = f'./Pretrained_models/model_29_1000_2l_154_256_noaug.pt'
    # Step 1: Load the pretrained model
    pretrained_model = torch.load(pretrained_path, map_location='cpu')["Teacher"]

    model = Model4TSNE(input_size=5, max_len=1000, embedding_size=154, track_size=14, hidden_size=256, mlp_dropout=0, layer_dropout=0, prenorm='None', norm='None')

    # for k, v in pretrained_model.items():
    #     print(k, v)

    # Step 2: Extract the encoder
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in pretrained_model.items():
        if k.startswith('encoder') or k.startswith('embedding'):
            print("*********************************************")
            new_state_dict[k] = v

    net_dict = model.state_dict()
    pretrained_cm = {k: v for k, v in new_state_dict.items() if k in net_dict}

    net_dict.update(pretrained_cm)
    model.load_state_dict(net_dict)

    model = model.cuda()

    # print("................", torch.cuda.memory_allocated())

    # for k, v in model.state_dict().items():
    #     print(k, v)

    # Step 4: Perform inference with the encoder
    train_X = torch.load(f"./data/{task}_X_train.pt")
    train_y = torch.load(f"./data/{task}_y_train.pt")
    test_X = torch.load(f"./data/{task}_X_test.pt")
    test_y = torch.load(f"./data/{task}_y_test.pt")

    print(train_X.shape)

    # train_X = cls_augment(train_X, 10)
    # test_X = cls_augment(test_X, 10)
    train_dataset = CustomDataset(train_X)
    test_dataset = CustomDataset(test_X)

    # Define batch sizes for training and testing
    batch_size = 64  # You can adjust this to your preferred batch size

    # Create DataLoader for training data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Create DataLoader for test data
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_output = []
    test_output = []
    
    with torch.no_grad():
      model.eval()
      for x in train_loader:
        output = model(x.cuda())
        train_output.extend(output)

      for x in test_loader:
        output = model(x.cuda())
        test_output.extend(output)

    train_X = torch.stack(train_output, dim=0)
    test_X = torch.stack(test_output, dim=0)

    # print(train_X.shape)
    # with torch.no_grad():
    #   model.eval()
    #   train_X = model(train_X.cuda())
    #   test_X = model(test_X.cuda())
    
    train_X = train_X.mean(dim=1)

    test_X = test_X.mean(dim=1)

    
    print(train_X.shape)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_X.cpu(), train_y.cpu())
    accuracy = knn.score(test_X.cpu(), test_y.cpu())
    print(f"KNN Classifier Accuracy: {accuracy}")