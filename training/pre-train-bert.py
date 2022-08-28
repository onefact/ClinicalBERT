import argparse
from util import str2bool, load_config
from pathlib import Path
import os
import wandb
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import torch
import numpy as np
from data.dataset import NotesforPT
from transformers import get_linear_schedule_with_warmup, BertTokenizerFast
from models.bert import BERTforPreTraining
from typing import Optional
import warnings
from dotenv import load_dotenv
from pytorch_lightning.callbacks import LearningRateMonitor

warnings.filterwarnings("ignore")


class PreTrainingModule(pl.LightningModule):
    def __init__(self, model, run_d):
        super().__init__()
        self.model = model
        self.run_d = run_d
        self.cross_loss = nn.CrossEntropyLoss()
        self.learning_rate = self.run_d["learning_rate"]
        self.save_hyperparameters()

    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        batches = len(self.trainer.datamodule.train_dataloader())
        x = batches * self.trainer.max_epochs / torch.cuda.device_count()
        return x

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        self.log("train_loss", outputs.loss)
        if batch_idx % 50 == 0:
            mlm_loss = self.cross_loss(
                outputs.prediction_logits.view(-1, outputs.prediction_logits.size(dim=2)), batch["labels"].view(-1)
            )
            nsp_loss = self.cross_loss(
                outputs.seq_relationship_logits.view(-1, 2), batch["next_sentence_label"].view(-1)
            )
            self.log("mlm_loss", mlm_loss)
            self.log("nsp_loss", nsp_loss)
        return outputs.loss

    def configure_optimizers(self):
        # set up optimizer
        if self.run_d["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.run_d["learning_rate"],
                betas=eval(self.run_d["betas"]),
                weight_decay=self.run_d["weight_decay"],
            )
        elif self.run_d["optimizer"] == "adamW":
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.run_d["learning_rate"], weight_decay=self.run_d["weight_decay"]
            )
        else:
            raise NotImplementedError

        # set up scheduler
        if self.run_d["scheduler"] == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, self.run_d["scheduler_period"], gamma=self.run_d["scheduler_ratio"]
            )
        elif self.run_d["scheduler"] == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=self.run_d["scheduler_period"], factor=self.run_d["scheduler_ratio"]
            )
        elif self.run_d["scheduler"] == "linear_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.num_training_steps() * 0.1,
                num_training_steps=self.num_training_steps(),
            )
        else:
            raise NotImplementedError

        print(f"Total training steps: {self.num_training_steps()}")

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_epoch_end(self, outputs):
        if self.trainer.current_epoch >= 0.9 * self.trainer.max_epochs:
            self.model.max_length = 512

        if self.trainer.is_global_zero and self.run_d["save_state"]:
            self.save_model()

    def save_model(self):
        mr_fp = os.path.join(wandb.run.dir, "bert-pre-train.pt")
        torch.save(self.model.state_dict(), mr_fp)


class PreTrainingDataModule(pl.LightningDataModule):
    def __init__(self, data_d, run_d, collate):
        super().__init__()
        self.data_d = data_d
        self.run_d = run_d
        self.collate = collate

    def setup(self, stage: Optional[str] = None) -> None:
        self.data = NotesforPT(self.data_d["data"])

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if self.trainer.current_epoch > 0.9 * self.trainer.max_epochs:
            return torch.utils.data.DataLoader(
                self.big_data,
                batch_size=self.run_d["big_batch_size"],
                shuffle=True,
                collate_fn=self.collate,
                pin_memory=True,
                num_workers=self.run_d["num_workers"],
            )
        else:
            return torch.utils.data.DataLoader(
                self.small_data,
                batch_size=self.run_d["small_batch_size"],
                shuffle=True,
                collate_fn=self.collate,
                pin_memory=True,
                num_workers=self.run_d["num_workers"],
            )


if __name__ == "__main__":

    load_dotenv()

    parser = argparse.ArgumentParser(description="Train on visits and code labels")
    parser.add_argument(
        "-t", "--template_fp", type=str, default="config/template.yaml", help="path to template config file"
    )
    parser.add_argument("-c", "--custom_fp", type=str, required=False, help="path to custom config file")
    flags = parser.parse_args()

    project_name, run_name, data_d, model_d, run_d = load_config(flags.template_fp, flags.custom_fp)
    Path(run_d['log_dir']).mkdir(parents=True, exist_ok=True)

    lr_monitor = LearningRateMonitor(logging_interval="step")

    wandb_logger = WandbLogger(project=project_name, name=run_name, save_dir=flags.log_dir)
    trainer = pl.Trainer(
        max_epochs=run_d["num_epochs"],
        gpus=torch.cuda.device_count(),
        logger=wandb_logger,
        strategy=DDPStrategy(find_unused_parameters=False),
        precision=16,
        profiler="simple",
        default_root_dir=run_d['log_dir'],
        callbacks=[lr_monitor],
    )

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model_dict = {"tokenizer": tokenizer, "max_length": model_d["max_length"], "config_file": model_d["config_file"]}
    model = BERTforPreTraining(**model_dict)

    data_module = PreTrainingDataModule(data_d, run_d, model.collate_fn)
    pre_training_module = PreTrainingModule(model, run_d)

    if 'load_state' in run_d:
        trainer.fit(pre_training_module, data_module, ckpt_path=run_d["chkpt_path"])
    else:
        trainer.fit(pre_training_module, data_module)
