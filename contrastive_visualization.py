import torch
import numpy as np
from torch import nn, optim
from omegaconf import OmegaConf
from functools import lru_cache
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from models.pretraining_model import Model4TSNE
from data_utils import vcf_Dataset
import matplotlib.pyplot as plt
import pytorch_lightning as pl    
from sklearn.manifold import TSNE
# from cuml.manifold import TSNE
from augment import RandomDeletion, RandomInsertion, RandomTranslocation
from transformers import get_cosine_schedule_with_warmup
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging, TQDMProgressBar
pl.seed_everything(42)


class LightningWrapper(pl.LightningModule):
    def __init__(self, model, cfg, snapshot_path, train_set, val_set, pretrained, loss, file_name):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model_config = self.hparams.SwanDNA
        self.batch_size = self.hparams.training.batch_size
        self.length = self.hparams.SwanDNA.max_len
        self.model = model(**self.model_config)#.apply(self._init_weights)
        self.save_every = self.hparams.training.save_every
        self.snapshot_path = snapshot_path
        self.train_set = train_set
        self.val_set = val_set
        self.loss = loss
        self.file_name = file_name

        print(self.model)

        if pretrained:
            pretrained_path = f'./Pretrained_models/{self.file_name}'
            pretrained = torch.load(pretrained_path, map_location='cpu')
            pretrained = pretrained["MODEL_STATE"]

            from collections import OrderedDict
            new_state_dict = OrderedDict()

            for k, v in pretrained.items():
                if k.startswith('encoder') or k.startswith('embedding'):
                    new_state_dict[k] = v

            net_dict = self.model.state_dict()
            pretrained_cm = {k: v for k, v in new_state_dict.items() if k in net_dict}

            net_dict.update(pretrained_cm)
            self.model.load_state_dict(net_dict)
            for k, v in self.model.state_dict().items():
                print(k, v)
            print("*************pretrained model loaded***************")

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.model(x)

    def _init_weights(self, m):
        if isinstance(m, nn.Reear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def training_step(self, batch, batch_idx):
        ref, alt, tissue, label = batch
        output = self.model(ref, alt, tissue).squeeze()
        train_loss = self.loss(output, label)
        return {"loss":train_loss, "preds":output, "labels":label, "tissue":tissue}

    def validation_step(self, batch, batch_idx):
        ref, alt, tissue, label = batch
        output = self.model(ref, alt, tissue).squeeze()
        val_loss = self.loss(output, label)
        return {"loss":val_loss, "preds":output, "labels":label, "tissue":tissue}


    def training_epoch_end(self, outputs):
        train_preds = [[] for _ in range(self.model_config.output_size)]
        train_labels = [[] for _ in range(self.model_config.output_size)]
        train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tissue = torch.stack([x["tissue"] for x in outputs]).reshape((-1,))
        label = torch.stack([x["labels"] for x in outputs]).reshape((-1,))
        output = torch.stack([x["preds"] for x in outputs]).reshape((-1,))

        for t, p, l in zip(tissue, output, label):
            t = t.to(torch.int8)
            train_preds[t.item()].append(p.item())
            train_labels[t.item()].append(l.item())
        train_rocs = []
        for i in range(self.model_config.output_size):
            rocauc = roc_auc_score(train_labels[i], train_preds[i])
            train_rocs.append(rocauc)
        train_roc = np.average(train_rocs)
        self.log('train_roc', train_roc, sync_dist=True)
        self.log('train_loss', train_loss, sync_dist=True)

    def validation_epoch_end(self, outputs):
        val_preds = [[] for _ in range(self.model_config.output_size)]
        val_labels = [[] for _ in range(self.model_config.output_size)]
        val_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tissue = torch.stack([x["tissue"] for x in outputs]).reshape((-1,))
        label = torch.stack([x["labels"] for x in outputs]).reshape((-1,))
        output = torch.stack([x["preds"] for x in outputs]).reshape((-1,))

        for t, p, l in zip(tissue, output, label):
            t = t.to(torch.int8)
            val_preds[t.item()].append(p.item())
            val_labels[t.item()].append(l.item())

        val_rocs = []
        for i in range(self.model_config.output_size):
            if len(val_labels[i]) != 0 and sum(val_labels[i]) != len(val_labels[i]) and sum(val_labels[i]) != 0:
                rocauc = roc_auc_score(val_labels[i], val_preds[i])
                val_rocs.append(rocauc)
        val_roc = np.average(val_rocs)
        self.log("val_auroc", val_roc, sync_dist=True)
        self.log('val_loss', val_loss, sync_dist=True)
        self.val_preds = [[] for _ in range(self.model_config.output_size)]
        self.val_labels = [[] for _ in range(self.model_config.output_size)]

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            num_workers=1,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            batch_size=self.batch_size
            )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            num_workers=1,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            batch_size=self.batch_size
            )

    @lru_cache
    def total_steps(self):
        l = len(self.trainer._data_connector._train_dataloader_source.dataloader())
        print('Num devices', self.trainer.num_devices)
        max_epochs = self.trainer.max_epochs
        accum_batches = self.trainer.accumulate_grad_batches
        manual_total_steps = (l // accum_batches * max_epochs)/self.trainer.num_devices
        print('MANUAL Total steps', manual_total_steps)
        return manual_total_steps

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.training.learning_rate,
            weight_decay=self.hparams.training.weight_decay
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=int(self.total_steps()*0.3),
                    num_training_steps=self.total_steps(),
                    num_cycles=self.hparams.training.n_cycles
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]


def classify_main(cfg):
    pretrained = cfg.Fine_tuning.training.pretrained
    length = cfg.Fine_tuning.SwanDNA.max_len

    loss = nn.BCEWithLogitsLoss()

    train_ref = torch.load(f"./data/ref_{length}_train.pt")
    train_alt = torch.load(f"./data/alt_{length}_train.pt")
    train_tissue = torch.load(f"./data/tissue_{length}_train.pt")
    train_label = torch.load(f"./data/label_{length}_train.pt")

    train_set =  vcf_Dataset(train_ref, train_alt, train_tissue, train_label)
    val_set = vcf_Dataset(torch.load(f"./data/ref_{length}_chr11_test.pt"), torch.load(f"./data/alt_{length}_chr11_test.pt"), torch.load(f"./data/tissue_{length}_chr11_test.pt"), torch.load(f"./data/label_{length}_chr11_test.pt"))

    ddp = DDPStrategy(process_group_backend="nccl", find_unused_parameters=True)
    snapshot_path = "test.pt"
    file_name = "SwanDNA_VE_10_16.pt"

    model = LightningWrapper(Classifier, cfg.Fine_tuning, snapshot_path, train_set, val_set, pretrained, loss, file_name)
    summary = ModelSummary(model, max_depth=-1)


    # ------------
    # init trainer
    # ------------

    wandb_logger = WandbLogger(dir="./wandb/", project="VE_classification", entity='', name=f'{file_name}_{length}_{pretrained}')
    checkpoint_callback = ModelCheckpoint(monitor="val_auroc", mode="max")

    print(len(train_set), len(val_set))

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks_for_trainer = [TQDMProgressBar(refresh_rate=10), lr_monitor, checkpoint_callback]
    if cfg.Fine_tuning.training.patience != -1:
        early_stopping = EarlyStopping(monitor="val_auroc", mode="max", min_delta=0., patience=cfg.Fine_tuning.training.patience)
        callbacks_for_trainer.append(early_stopping)
    if cfg.Fine_tuning.training.swa_lrs != -1:
        swa = StochasticWeightAveraging(swa_lrs=1e-2)
        callbacks_for_trainer.append(swa)

    print(summary)
    trainer = pl.Trainer(
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        accelerator='gpu',
        strategy=ddp,
        devices=[0],
        max_epochs=cfg.Fine_tuning.training.n_epochs,
        gradient_clip_val=0.5,
        num_sanity_val_steps=0,
        # profiler=profiler,
        precision=16,
        logger=wandb_logger
    )
    trainer.fit(model)

def cls_augment(masked_gene, local_cls_number):
    N, L, D = masked_gene.shape
    cls_masked = torch.zeros(N, local_cls_number, D)

    masked_gene = torch.cat((masked_gene, cls_masked), 1)
    return masked_gene


if __name__ == "__main__":
    pretrained_path = f'./Pretrained_models/model_10_1000_2l_154_256.pt'
    # Step 1: Load the pretrained model
    pretrained_model = torch.load(pretrained_path, map_location='cpu')["Teacher"]

    model = Model4TSNE(input_size=5, max_len=1000, embedding_size=154, track_size=14, hidden_size=256, mlp_dropout=0, layer_dropout=0, prenorm='None', norm='None')

    for k, v in pretrained_model.items():
        print(k, v)

    # Step 2: Extract the encoder
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in pretrained_model.items():
        if k.startswith('encoder') or k.startswith('embedding'):
            print("**************************************************************************************")
            new_state_dict[k] = v

    net_dict = model.state_dict()
    pretrained_cm = {k: v for k, v in new_state_dict.items() if k in net_dict}

    net_dict.update(pretrained_cm)
    model.load_state_dict(net_dict)

    for k, v in model.state_dict().items():
        print(k, v)

    # Step 4: Perform inference with the encoder
    genes_train = torch.load(f"./data/gene_valid_1000_10k.pt")
    augment_list = [
        RandomDeletion(delete_min=0, delete_max=20),
        RandomInsertion(insert_min=0, insert_max=20),
        RandomTranslocation(shift_min=0, shift_max=20)
    ]
    for augment in augment_list:
        genes_train_aug = torch.permute(augment(torch.permute(genes_train, (0, 2, 1))), (0, 2, 1))
    
    genes_train_aug = cls_augment(genes_train_aug, 10)

    with torch.no_grad():
        x = model(genes_train_aug)
        print(x.shape)
    
    x = x[:, 1000:, :]
    x = x.view(x.shape[0], -1)

    # x = x[:, :1000, :]
    # x = x.mean(dim=1).view(x.shape[0], -1)
    print(x.shape)

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(x)
    fig = plt.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], marker='.', s=2)
    plt.savefig("teacher_tsne_cls_10k_154_2l_E20.png")
    plt.show()

