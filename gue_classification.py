import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from omegaconf import OmegaConf
from functools import lru_cache
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MatthewsCorrCoef, F1Score
from torchmetrics.classification import MulticlassMatthewsCorrCoef
from models.SwanDNA import GB_Flash_Classifier, GB_Linear_Classifier
from data_utils import gb_Dataset
# from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging, TQDMProgressBar
pl.seed_everything(42)


class LightningWrapper(pl.LightningModule):
    def __init__(self, model, cfg, train_set, val_set, test_set, pretrained, loss, file_name):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model_config = self.hparams.SwanDNA
        self.batch_size = self.hparams.training.batch_size
        self.output = self.hparams.SwanDNA.output_size
        self.warm_up = self.hparams.training.n_warmup_steps
        self.length = self.hparams.SwanDNA.max_len
        self.model = model(**self.model_config)
        self.save_every = self.hparams.training.save_every
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.loss = loss
        self.file_name = file_name
        # if self.output == 2:
        #     self.train_mcc = MatthewsCorrCoef(task='binary')
        #     self.val_mcc = MatthewsCorrCoef(task='binary')
        #     self.test_mcc = MatthewsCorrCoef(task='binary')
        # else:
        #     self.train_mcc = MulticlassMatthewsCorrCoef(num_classes=3)
        #     self.val_mcc = MulticlassMatthewsCorrCoef(num_classes=3)
        #     self.test_mcc = MulticlassMatthewsCorrCoef(num_classes=3)

        if self.hparams.training.name == "virus":
            self.train_mcc = F1Score(task="multiclass", num_classes=9)
            self.val_mcc = F1Score(task="multiclass", num_classes=9)
            self.test_mcc = F1Score(task="multiclass", num_classes=9)
        print(self.model)

        if pretrained:
            pretrained_path = f'./{self.file_name}'
            pretrained = torch.load(pretrained_path, map_location='cpu')
            pretrained = pretrained["Teacher"]

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
            print(self.file_name)
            print("*************pretrained model loaded***************")


    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.model(x)

    def _init_weights(self, m):
        if isinstance(m, nn.Reear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def training_step(self, batch, batch_idx):
        seq, label = batch
        output = self.model(seq).squeeze()
        preds = output.argmax(dim=-1)
        train_loss = self.loss(output, label.to(torch.int64))
        self.train_mcc.update(preds, label.int())
        return {"loss":train_loss, "preds":preds, "labels":label}

    def validation_step(self, batch, batch_idx):
        seq, label = batch
        output = self.model(seq).squeeze()
        preds = output.argmax(dim=-1)
        val_loss = self.loss(output, label.to(torch.int64))
        self.val_mcc.update(preds, label.int())
        return {"loss":val_loss, "preds":preds, "labels":label}
    
    def test_step(self, batch, batch_idx):
        seq, label = batch
        output = self.model(seq).squeeze()
        preds = output.argmax(dim=-1)
        test_loss = self.loss(output, label.to(torch.int64))
        self.test_mcc.update(preds, label.int())
        return {"loss":test_loss, "preds":preds, "labels":label}

    def training_epoch_end(self, outputs):
        train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        acc = self.train_mcc.compute().mean()
        self.train_mcc.reset()
        self.log('train_mcc', acc, sync_dist=True)
        self.log('train_loss', train_loss, sync_dist=True)

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["loss"] for x in outputs]).mean()
        # label = torch.stack([x["labels"] for x in outputs]).reshape((-1,))
        # output = torch.stack([x["preds"] for x in outputs]).reshape((-1,))
        acc = self.val_mcc.compute().mean()
        self.val_mcc.reset()
        self.log("val_mcc", acc, sync_dist=True)
        self.log('val_loss', val_loss, sync_dist=True)

    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x["loss"] for x in outputs]).mean()
        # label = torch.stack([x["labels"] for x in outputs]).reshape((-1,))
        # output = torch.stack([x["preds"] for x in outputs]).reshape((-1,))
        acc = self.test_mcc.compute().mean()
        self.val_mcc.reset()
        self.log("test_mcc", acc, sync_dist=True)
        self.log('test_loss', test_loss, sync_dist=True)

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
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
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
                    num_warmup_steps=int(self.total_steps()*self.warm_up), #hyperparmeter [0.3, 0.4] 
                    num_training_steps=self.total_steps(),
                    num_cycles=self.hparams.training.n_cycles
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
    

def sequence2onehot(data_file, lb, length):
    ds = pd.read_csv(data_file)
    sequences, labels = [],[]
    for index, data in ds.iterrows():
        gene_to_number = lb.transform(list(data["sequence"]))
        if gene_to_number.shape[0] == length:
            sequences.append(gene_to_number)
            labels.append(data["label"])
    X = torch.from_numpy(np.array(sequences)).to(torch.int8)
    y = torch.from_numpy(np.array(labels)).to(torch.float16)

    return X, y


def classify_main(cfg, task, branch):
    """
    1. decide which tack to run
    """
    if task == "H3":
        config = cfg.H3
    elif task == "H3K4me1":
        config = cfg.H3K4me1
    elif task == "H3K4me2":
        config = cfg.H3K4me2
    elif task == "H3K4me3":
        config = cfg.H3K4me3
    elif task == "H3K36me3":
        config = cfg.H3K36me3
    elif task == "H3K14ac":
        config = cfg.H3K14ac
    elif task == "H4":
        config = cfg.H4
    elif task == "H3K79me3":
        config = cfg.H3K79me3
    elif task == "H3K9ac":
        config = cfg.H3K9ac
    elif task == "H4ac":
        config = cfg.H4ac
    elif task == "prom_core_notata":
        config = cfg.Prom_notata
    elif task == "prom_core_tata":
        config = cfg.Prom_tata
    elif task == "prom_core_all":
        config = cfg.Prom_all
    elif task == "prom_300_notata":
        config = cfg.Prom_300_notata
    elif task == "prom_300_tata":
        config = cfg.Prom_300_tata
    elif task == "prom_300_all":
        config = cfg.Prom_300_all
    elif task == "tf1":
        config = cfg.tf1
    elif task == "tf3":
        config = cfg.tf3
    elif task == "splice":
        config = cfg.Splice
    elif task == "virus":
        config = cfg.virus

    
    """
    2. load dataset.
    """

    pretrained = config.training.pretrained
    length = config.SwanDNA.max_len
    loss = nn.CrossEntropyLoss(reduction='mean')

    lb = LabelBinarizer()
    lb.fit(['A', 'T', 'C', 'G', 'N'])

    df = pd.read_csv(f"./data/GUE/GUE/virus/{branch}/train.csv")
    print(df.describe())

    train_X, train_y = sequence2onehot(f"./data/GUE/GUE/virus/{branch}/train.csv", lb, length)
    val_X, val_y = sequence2onehot(f"./data/GUE/GUE/virus/{branch}/dev.csv", lb, length)
    test_X, test_y = sequence2onehot(f"./data/GUE/GUE/virus/{branch}/test.csv", lb, length)
    print("***************data******************")
    # print(train_X[0])
    print(train_X.size(), test_X.size(), val_X.size())

    train_set =  gb_Dataset(train_X, train_y)
    val_set = gb_Dataset(val_X, val_y)
    test_set = gb_Dataset(test_X, test_y)

    test_dalaloader = DataLoader(
            dataset=test_set,
            num_workers=1,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            batch_size=config.training.batch_size
            )

    """
    3. strat training with ddp mode.
    """

    ddp = DDPStrategy(process_group_backend="nccl", find_unused_parameters=True)
    pretrained_model = "model_29_1000_4l_308_512_noiseandTL.pt"

    model = LightningWrapper(GB_Linear_Classifier, config, train_set, val_set, test_set, pretrained, loss, pretrained_model)
    summary = ModelSummary(model, max_depth=-1)

    """
    4. init trainer
    """

    wandb_logger = WandbLogger(dir="./wandb/", project="Prom", entity='tonyu', name=f'{pretrained_model}_{length}_{branch}')
    checkpoint_callback = ModelCheckpoint(monitor="val_mcc", mode="max")

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks_for_trainer = [TQDMProgressBar(refresh_rate=10), lr_monitor, checkpoint_callback]
    if config.training.patience != -1:
        early_stopping = EarlyStopping(monitor="val_mcc", mode="max", min_delta=0., patience=cfg.Fine_tuning.training.patience)
        callbacks_for_trainer.append(early_stopping)
    if config.training.swa_lrs != -1:
        swa = StochasticWeightAveraging(swa_lrs=1e-2)
        callbacks_for_trainer.append(swa)

    print(summary)
    trainer = pl.Trainer(
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        accelerator='gpu',
        strategy=ddp,
        devices=[0],
        max_epochs=config.training.n_epochs,
        gradient_clip_val=0.5,
        num_sanity_val_steps=0,
        precision=16,
        logger=wandb_logger,
        callbacks=callbacks_for_trainer
    )
    trainer.fit(model)

    trainer.test(model, test_dalaloader, "best")


if __name__ == "__main__":
    cfg = OmegaConf.load('./config/config_gue.yaml')
    OmegaConf.set_struct(cfg, False)
    classify_main(cfg, "virus", "covid")
