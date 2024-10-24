from random import random as rand
from evoaug import evoaug, augment
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
import numpy as np
from functools import lru_cache
import pytorch_lightning as pl
from proselflc import ProSelfLC
from torch import nn, optim
from augment import RandomDeletion, RandomInsertion, RandomTranslocation, RandomNoise, RandomRC
from transformers import get_cosine_schedule_with_warmup
from models.pretraining_model import Model4Pretrain, Model4PretrainFlash
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging, TQDMProgressBar
pl.seed_everything(42)


def pretrain_loss(loss, preds, labels, masks):
    masks_new = masks.repeat(5, 1, 1)#.reshape(preds.shape)
    # print("losssssss", masks_new.shape, preds.shape, labels.shape)
    masks_new = torch.reshape(masks_new, preds.shape)

    print(labels[0][0:10])
    print(preds[0][0:10])

    labels = labels[masks_new == 1]
    preds = preds[masks_new == 1]

    return loss(preds.float(), labels.float())


class proselfLCLoss(nn.Module):
    def __init__(self, params, out_dim, patch_out_dim, ngcrops, nlcrops, warmup_teacher_temp, 
                 teacher_temp, warmup_teacher_temp2, teacher_temp2, 
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, 
                 center_momentum=0.9, center_momentum2=0.9,
                 lambda1=1, lambda2=1, mim_start_epoch=0, length=1000):
        super().__init__()
        self.params = params
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ngcrops = ngcrops
        self.nlcrops = nlcrops
        self.ncrops = ngcrops + nlcrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, out_dim))
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.none_cls_length = length

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.teacher_temp2_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp2
        )) if mim_start_epoch == 0 else np.concatenate((
            np.ones(mim_start_epoch) * warmup_teacher_temp2,
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs - mim_start_epoch) * teacher_temp2
        ))
        

    def forward(self, student_output, teacher_output, student_local_cls, student_mask, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # print(student_output[0].shape)
        student_patch = student_output[0][:,0: self.none_cls_length,:]
        student_cls = student_output[0][:,self.none_cls_length:,:]
        teacher_patch = teacher_output[:, 0:self.none_cls_length, :]
        teacher_cls = teacher_output[:, self.none_cls_length:,:]

        # print("*******", student_cls.shape, student_patch.shape)
        
        if student_local_cls is not None:
            student_cls = torch.cat([student_cls, student_local_cls])

        # [CLS] and patch for global patches
        student_cls = student_cls / self.student_temp
        # student_cls_c = student_cls.chunk(self.ncrops)
        student_patch = student_patch / self.student_temp
        # student_patch_c = student_patch.chunk(self.ngcrops)
        
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        # print(teacher_cls.shape, self.center.shape)
        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach()
        teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp2, dim=-1)
        teacher_patch_c = teacher_patch_c.detach()

        print("**********************", teacher_cls_c.shape, student_cls.shape)

        total_loss1 = 0
        total_loss2 = 0

        # total_loss1 = torch.sum(-teacher_cls_c * F.log_softmax(student_cls, dim=-1), dim=-1).mean()
        proselflc_loss = ProSelfLC(self.params)
        total_loss1 = proselflc_loss(F.softmax(student_cls, dim = -1), F.softmax(teacher_cls_c, dim = -1), epoch)

        print("*****************", total_loss1, total_loss2)

        loss_func = nn.BCEWithLogitsLoss(reduction='mean')

        total_loss2 = pretrain_loss(loss_func, student_patch, student_output[1], student_output[2])
        # total_loss2 = ProSelfLC(params, )
            
        total_loss1 = total_loss1 * self.lambda1
        total_loss2 = total_loss2 * self.lambda2
        print("loss1", total_loss1, "loss2", total_loss2)
        total_loss = dict(cls=total_loss1, patch=total_loss2, loss=total_loss1 + total_loss2)
        # total_loss = dict(cls=total_loss1, patch=total_loss2, loss=total_loss2)
        self.update_center(teacher_cls)

        return total_loss
    
    @torch.no_grad()
    def update_center(self, teacher_cls):
        """
        Update center used for teacher output.
        """
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        # dist.all_reduce(cls_center)
        cls_center = cls_center / len(teacher_cls) # * dist.get_world_size())
        self.center = self.center * self.center_momentum + cls_center * (1 - self.center_momentum)



class DatasetCreator(Dataset):
    """
    Class to construct a dataset for training/inference
    """
    def __init__(self, original_gene, augmented_genes, masked_genes, masks):
        self.genes = original_gene
        self.augmented_genes = augmented_genes
        self.masked_genes = masked_genes
        self.masks = masks

    def __getitem__(self, index):
        return (self.genes[index], self.augmented_genes[index], self.masked_genes[index], self.masks[index])

    def __len__(self):
        return len(self.genes)


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class LightningWrapper(pl.LightningModule):
    def __init__(self, model, cfg, snapshot_path, train_set, val_set, loss):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model_config = self.hparams.training
        self.arch_config = self.hparams.SwanDNA
        self.batch_size = self.hparams.training.batch_size
        self.length = self.model_config.max_len
        self.student = model(**self.arch_config)
        self.teacher = model(**self.arch_config)
        self.teacher.load_state_dict(self.student.state_dict(), strict=False)
        self.save_every = self.hparams.training.save_every
        self.snapshot_path = snapshot_path
        self.train_set = train_set
        self.val_set = val_set
        self.params = {
            "total_epochs": self.model_config.n_epochs,
            "exp_base": 6,
            "counter": "epoch",
            "transit_time_ratio": 0.3
        }
        self.loss = proselfLCLoss(
            self.params,
            self.model_config.out_dim,
            self.model_config.out_dim,
            self.model_config.global_crops_number,
            self.model_config.local_crops_number,
            self.model_config.warmup_teacher_temp,
            self.model_config.teacher_temp,
            self.model_config.warmup_teacher_patch_temp,
            self.model_config.teacher_patch_temp,
            self.model_config.warmup_teacher_temp_epochs,
            self.model_config.n_epochs,
            lambda1=self.model_config.lambda1,
            lambda2=self.model_config.lambda2,
            mim_start_epoch=self.model_config.pred_start_epoch,
            length=self.model_config.max_len
        )
        self.momentum_schedule = cosine_scheduler(0.996, 1, self.model_config.n_epochs, len(self.train_dataloader()))

        for p in self.teacher.parameters():
            p.requires_grad = False

        print(self.student, self.teacher)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.model(x)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def training_step(self, batch, batch_idx):
        # common params
        names_q, params_q, names_k, params_k = [], [], [], []
        for name_q, param_q in self.student.state_dict().items():
            names_q.append(name_q)
            params_q.append(param_q)
        for name_k, param_k in self.teacher.state_dict().items():
            names_k.append(name_k)
            params_k.append(param_k)
        names_common = list(set(names_q) & set(names_k))
        params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
        params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]

        original_gene, augmented_gene, masked_gene, masks = batch
        # print("origin", original_gene.shape)
        # get global views
        teacher_output = self.teacher(augmented_gene)
        student_output = [self.student(masked_gene), original_gene, masks]

        # get local views
        # self.student.module.backbone.masked_im_modeling = False
        # student_local_cls = self.student(masked_gene[self.model_config.global_crops_number:])[0] if len(masked_gene) > self.model_config.global_crops_number else None
        student_local_cls = None
        # self.student.module.backbone.masked_im_modeling = self.model_config.use_masked_im_modeling

        all_loss = self.loss(student_output, teacher_output, student_local_cls, masks, self.current_epoch)
        loss = all_loss.pop('loss')
        cls_loss = all_loss.pop('cls')
        mlm_loss = all_loss.pop('patch')

        with torch.no_grad():
            # m = self.optimizers().param_groups[0]['lr']#/self.model_config.learning_rate  # momentum parameter
            m = self.momentum_schedule[self.global_step]
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        self.log('train_loss', loss, sync_dist=True)
        self.log('cls_loss', cls_loss, sync_dist=True)
        self.log('mlm_loss', mlm_loss, sync_dist=True)

        return {"loss":loss}
    
    def training_epoch_end(self, outputs):
        if self.current_epoch ==9 or self.current_epoch == self.model_config.n_epochs-1:
            self._save_snapshot()

    def validation_step(self, batch, batch_idx):
        original_gene, augmented_gene, masked_gene, masks = batch
        # get global views
        teacher_output = self.teacher(augmented_gene)
        student_output = [self.student(masked_gene), original_gene, masks]
        
        # get local views
        # student_local_cls = self.student(masked_gene[self.model_config.global_crops_number:])[0] if len(masked_gene) > self.model_config.global_crops_number else None

        student_local_cls = None

        all_loss = self.loss(student_output, teacher_output, student_local_cls, masks, self.current_epoch)
        loss = all_loss.pop('loss')
        cls_loss = all_loss.pop('cls')
        mlm_loss = all_loss.pop('patch')

        return {"loss":loss, "cls_loss":cls_loss, "mlm_loss":mlm_loss}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["loss"] for x in outputs]).mean()
        val_cls_loss = torch.stack([x["cls_loss"] for x in outputs]).mean()
        val_mlm_loss = torch.stack([x["mlm_loss"] for x in outputs]).mean()
        self.log('val_loss', val_loss, sync_dist=True)
        self.log('val_cls_loss', val_cls_loss, sync_dist=True)
        self.log('val_mlm_loss', val_mlm_loss, sync_dist=True)

    def _save_snapshot(self):
        snapshot = {
            "Teacher": self.teacher.state_dict(),
            "Student": self.student.state_dict(),
            "EPOCHS_RUN": self.current_epoch ,
        }
        torch.save(snapshot, f"{self.snapshot_path}/proselflc_{self.current_epoch}_{self.length}_4l_308_512_noiseandTL.pt")
        print(f"Epoch {self.current_epoch } | Training snapshot saved at {self.snapshot_path}")

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:0"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

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
            drop_last=True,
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
                    num_warmup_steps=self.total_steps()*0.3,
                    num_training_steps=self.total_steps(),
                    num_cycles=self.hparams.training.n_cycles
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
    
# def cls_augment(gene, masked_gene, local_cls_number):
#     N, L, D = gene.shape
#     # random_masks = torch.zeros(local_cls_number, L)
#     # cls_masked = np.eye(D)[np.random.randint(0, D, (N, local_cls_number, 1))].squeeze()
#     cls_masked = torch.zeros(N, local_cls_number, D)
#     # cls = np.eye(D)[np.random.randint(0, D, (N, local_cls_number, 1))].squeeze()
#     cls = torch.zeros(N, local_cls_number, D)

#     gene = torch.cat((cls, gene), 1)
#     masked_gene = torch.cat((cls_masked, masked_gene), 1)
#     return gene, masked_gene

def cls_augment(masked_gene, local_cls_number):
    N, L, D = masked_gene.shape
    cls_masked = torch.zeros(N, local_cls_number, D)

    masked_gene = torch.cat((masked_gene, cls_masked), 1)
    return masked_gene


def pretrain_main(cgf):
    """
    # 1. Load data for pretraining
    """
    genes_train = torch.load(f"./data/gene_train_{cfg.Pretraining.training.max_len}_100k.pt")
    masked_genes_train = torch.load(f"./data/masked_train_{cfg.Pretraining.training.max_len}_100k.pt")
    masks_train = torch.load(f"./data/mask_train_{cfg.Pretraining.training.max_len}_100k.pt")

    genes_val = torch.load(f"./data/gene_val_{cfg.Pretraining.training.max_len}_100k.pt")
    masked_genes_val = torch.load(f"./data/masked_val_{cfg.Pretraining.training.max_len}_100k.pt")
    masks_val = torch.load(f"./data/mask_val_{cfg.Pretraining.training.max_len}_100k.pt")
    
    print(genes_train.shape, genes_val.shape)
    print(genes_train[0][0:10], masked_genes_train[0][0:10], masks_train[0][0:10])
    original_train = genes_train
    original_val = genes_val


    # 2. Augment the Data
    # 2.1   Add CLS tokens

    
    print(genes_train.shape, masked_genes_train.shape, masks_train.shape)

    augment_list_1 = [
        RandomDeletion(delete_min=0, delete_max=20),
        RandomInsertion(insert_min=0, insert_max=20),
        RandomTranslocation(shift_min=0, shift_max=20)
    ]
    for augment in augment_list_1:
        genes_train_aug = torch.permute(augment(torch.permute(genes_train, (0, 2, 1))), (0, 2, 1))
    
    for augment in augment_list_1:
        genes_val_aug = torch.permute(augment(torch.permute(genes_val, (0, 2, 1))), (0, 2, 1))

    # genes_train_aug = genes_train
    # genes_val_aug = genes_val


    augment_list_2 = [
        # RandomDeletion(delete_min=0, delete_max=20),
        # RandomInsertion(insert_min=0, insert_max=20),
        RandomNoise(0, 0.2),
        # RandomTranslocation(shift_min=0, shift_max=20)
        RandomRC(0.5)
    ]

    for augment in augment_list_2:
        masked_genes_train = torch.permute(augment(torch.permute(masked_genes_train, (0, 2, 1))), (0, 2, 1))
    
    for augment in augment_list_2:
        masked_genes_val = torch.permute(augment(torch.permute(masked_genes_val, (0, 2, 1))), (0, 2, 1))

    print("masked after augmentation", masked_genes_train.shape)

    masked_genes_train = cls_augment(masked_genes_train, 10)
    masked_genes_val = cls_augment(masked_genes_val, 10)

    # print("before" ,genes_train_aug.shape)
    genes_train_aug = cls_augment(genes_train_aug, 10)
    genes_val_aug = cls_augment(genes_val_aug, 10)

    print(genes_train_aug.shape, masked_genes_train.shape)

    # import torch.nn.functional as F

    # dataset1_flat = genes_train_aug.view(-1, 5)
    # dataset2_flat = masked_genes_train.view(-1, 5)

    # print(dataset1_flat[:5])
    

    # # Apply a softmax to make sure each row is a valid probability distribution
    # dataset1_probs = F.softmax(dataset1_flat, dim=1)
    # dataset2_probs = F.softmax(dataset2_flat, dim=1)

    # print(dataset2_probs[:5])

    # # Calculate KL divergence
    # kl_divergence = F.kl_div(torch.log(dataset1_probs), dataset2_probs, reduction='batchmean')

    # print("KL Divergence:", kl_divergence.item())

    # sys.exit(-1)

    print("after cls augmentation", masked_genes_train.shape, genes_train_aug.shape)

    # genes_train_aug = cls_augment(original_train, 10)
    # genes_val_aug = cls_augment(original_val, 10)

    # print("after", genes_train_aug.shape, original_train.shape)

    train_set = DatasetCreator(original_train, genes_train_aug, masked_genes_train, masks_train)
    val_set =  DatasetCreator(original_val, genes_val_aug, masked_genes_val, masks_val)

    """
    # 3. Prepare model
    """

    ddp = DDPStrategy(process_group_backend="nccl", find_unused_parameters=True)
    # profiler = SimpleProfiler()
    snapshot_path = "./Pretrained_models/"

    # loss = nn.CrossEntropyLoss(reduce="sum")
    loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
    MetaArch =  Model4Pretrain
    model = LightningWrapper(MetaArch, cfg.Pretraining, snapshot_path, train_set, val_set, loss)
    print(model)
    summary = ModelSummary(model, max_depth=-1)
    wandb_logger = WandbLogger(dir="./wandb/", project="Contrastive_Pretrain", entity='tonyu', name=f'ProselfLC_{cfg.Pretraining.training.max_len}_4l_{cfg.Pretraining.SwanDNA.embedding_size}_{cfg.Pretraining.SwanDNA.hidden_size}')
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks_for_trainer = [TQDMProgressBar(refresh_rate=10), lr_monitor, checkpoint_callback]
    
    """
    # 4. init trainer
    """

    print(summary)
    trainer = pl.Trainer(
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        accelerator='gpu',
        strategy=ddp,
        devices=[0, 1],
        max_epochs=cfg.Pretraining.training.n_epochs,
        gradient_clip_val=0.5,
        num_sanity_val_steps=0,
        precision=16,
        logger=wandb_logger,
        callbacks=callbacks_for_trainer
    )
    trainer.fit(model)


if __name__ == "__main__":
    cfg = OmegaConf.load('./config/config_ct.yaml') #for ve pretraining, chenge it to config.yaml
    OmegaConf.set_struct(cfg, False)
    pretrain_main(cfg)
