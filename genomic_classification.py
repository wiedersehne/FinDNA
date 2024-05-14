import torch
from torch import nn, optim
from omegaconf import OmegaConf
from functools import lru_cache
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
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
    def __init__(self, model, cfg, train_set, val_set, pretrained, loss, file_name):
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
        self.loss = loss
        self.file_name = file_name
        if self.output == 2:
            self.train_acc = Accuracy(task='binary', top_k=1)
            self.val_acc = Accuracy(task='binary', top_k=1)
        else:
            self.train_acc = Accuracy(task='multiclass', num_classes=self.model_config.output_size, top_k=1)
            self.val_acc = Accuracy(task='multiclass', num_classes=self.model_config.output_size, top_k=1)
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
        self.train_acc.update(preds, label.int())
        return {"loss":train_loss, "preds":preds, "labels":label}

    def validation_step(self, batch, batch_idx):
        seq, label = batch
        output = self.model(seq).squeeze()
        preds = output.argmax(dim=-1)
        val_loss = self.loss(output, label.to(torch.int64))
        self.val_acc.update(preds, label.int())
        return {"loss":val_loss, "preds":preds, "labels":label}

    def training_epoch_end(self, outputs):
        train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        acc = self.train_acc.compute().mean()
        self.train_acc.reset()
        self.log('train_acc', acc, sync_dist=True)
        self.log('train_loss', train_loss, sync_dist=True)

    # def validation_step_end(self, outputs):
    #     acc = self.val_acc(outputs["preds"], outputs["labels"])
    #     self.log("val_acc", acc, sync_dist=True)
    #     self.log('val_loss', outputs["loss"], sync_dist=True)

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["loss"] for x in outputs]).mean()
        # label = torch.stack([x["labels"] for x in outputs]).reshape((-1,))
        # output = torch.stack([x["preds"] for x in outputs]).reshape((-1,))
        acc = self.val_acc.compute().mean()
        self.val_acc.reset()
        self.log("val_acc", acc, sync_dist=True)
        self.log('val_loss', val_loss, sync_dist=True)


    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            num_workers=1,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
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
                    num_warmup_steps=int(self.total_steps()*self.warm_up), #hyperparmeter [0.3, 0.4] 
                    num_training_steps=self.total_steps(),
                    num_cycles=self.hparams.training.n_cycles
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]


def classify_main(cfg, task):
    """
    1. decide which tack to run
    """
    if task == "human_nontata_promoters":
        config = cfg.Human_Promoter
    elif task == "human_enhancers_cohn":
        config = cfg.Human_Enhancers_Cohn
    elif task == "demo_human_or_worm":
        config = cfg.Demo_Human_Or_Worm
    elif task == "dummy_mouse_enhancers_ensembl":
        config = cfg.Demo_Mouse_Enhancers
    elif task == "demo_coding_vs_intergenomic_seqs":
        config = cfg.Demo_Coding_Inter
    elif task == "drosophila_enhancers_stark":
        config = cfg.Drop_Enhancer_Stark
    elif task == "human_enhancers_ensembl":
        config = cfg.Human_Enhancers_Ensembl
    elif task == "human_ensembl_regulatory":
        config = cfg.Human_Regulatory
    elif task == "human_ocr_ensembl":
        config = cfg.Human_Ocr_Ensembl
    
    """
    2. load dataset.
    """

    pretrained = config.training.pretrained
    length = config.SwanDNA.max_len
    loss = nn.CrossEntropyLoss(reduction='mean')

    train_X = torch.load(f"./data/{task}_X_train.pt")
    train_y = torch.load(f"./data/{task}_y_train.pt")
    test_X = torch.load(f"./data/{task}_X_test.pt")
    test_y = torch.load(f"./data/{task}_y_test.pt")
    print(train_X.shape)

    train_set =  gb_Dataset(train_X, train_y)
    val_set = gb_Dataset(test_X, test_y)

    """
    3. strat training with ddp mode.
    """

    ddp = DDPStrategy(process_group_backend="nccl", find_unused_parameters=True)
    pretrained_model = "model_29_1000_4l_308_512_noiseandTL.pt"

    model = LightningWrapper(GB_Linear_Classifier, config, train_set, val_set, pretrained, loss, pretrained_model)
    summary = ModelSummary(model, max_depth=-1)

    """
    4. init trainer
    """

    wandb_logger = WandbLogger(dir="./wandb/", project="Mouse_Enhancers", entity='tonyu', name=f'{pretrained_model}_{length}_{task}')
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max")

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks_for_trainer = [TQDMProgressBar(refresh_rate=10), lr_monitor, checkpoint_callback]
    if config.training.patience != -1:
        early_stopping = EarlyStopping(monitor="val_acc", mode="max", min_delta=0., patience=cfg.Fine_tuning.training.patience)
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
        logger=wandb_logger
    )
    trainer.fit(model)


if __name__ == "__main__":
    cfg = OmegaConf.load('./config/config_gb.yaml')
    OmegaConf.set_struct(cfg, False)
    classify_main(cfg, "human_ocr_ensembl")
