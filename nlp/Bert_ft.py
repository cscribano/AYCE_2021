# -*- coding: utf-8 -*-
# ---------------------

import os
import json
import random
import click

import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from transformers import BertTokenizer, BertModel
from transformers import AdamW

class NL_Dataset(Dataset):

	def __init__(self, ds_root, mode="train"):

		assert mode in ["train", "val"]
		self.ds_root = ds_root
		self.mode = mode

		# load train/val keys
		mode_files = {"train": "../data/train.txt", "val": "../data/validation.txt"}
		split_file = os.path.join(os.path.dirname(__file__), mode_files[mode])

		# read split uuids
		with open(split_file) as f:
			self.uuids = [line.rstrip() for line in f]
			f.close()

		# Load train split json
		tracks_root = os.path.join(ds_root, 'data/train-tracks.json')
		with open(tracks_root, "r") as f:
			tracks = json.load(f)
			f.close()

		self.tracks = tracks

	def __len__(self):
		# type: () -> int
		return len(self.uuids)

	def __getitem__(self, item):
		# type: (int) -> (tuple, tuple)

		uuid = self.uuids[item]
		pos = self.tracks[uuid]["nl"]  # positive
		np = torch.randperm(3)

		positive = pos[np[0]]
		anchor = pos[np[1]]

		# pick uuid of negative embedding
		if self.mode != "val":
			# For negative samples pick a random one
			ix = item
			while ix == item:
				# Avoid picking the real index
				ix = random.randint(0, len(self.uuids) - 1)
			negative_uuid = self.uuids[ix]
		else:
			# Make validation deterministic
			negative_uuid = self.uuids[(item ** 2 + 8) % len(self.uuids)]

		# negative
		neg = self.tracks[negative_uuid]["nl"]  # positive (3)
		np = torch.randperm(3)
		negative = neg[np[0]]

		return anchor, positive, negative

class LiteBert(pl.LightningModule):
	def __init__(self, learning_rate=1e-4):
		super().__init__()
		self.save_hyperparameters()
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
		self.bert_model = BertModel.from_pretrained('bert-base-uncased')

		# loss
		self.criterion = torch.nn.TripletMarginLoss(margin=2.5)

	def forward(self, x):
		# use forward for inference/predictions
		tokens = self.tokenizer(x, padding='longest', return_tensors='pt')
		mask = tokens['attention_mask'].to(self.device)
		bert_out = self.bert_model(tokens['input_ids'].to(self.device),
								   attention_mask=mask).last_hidden_state

		bert_out = torch.mean(bert_out, dim=1)

		return bert_out  # (BS, 3, 768)

	def training_step(self, batch, batch_idx):
		anchor, pos, neg = batch
		anchor_emb = self.forward(anchor)
		pos_emb = self.forward(pos)
		neg_emb = self.forward(neg)

		loss = self.criterion(anchor_emb, pos_emb, neg_emb)
		self.log('train_loss', loss, on_epoch=True)
		return loss

	def validation_step(self, batch, batch_idx):
		loss = self.training_step(batch, batch_idx)
		self.log('valid_loss', loss, on_epoch=True)
		return loss


	def configure_optimizers(self):
		# self.hparams available because we called self.save_hyperparameters()
		return AdamW(self.parameters(), lr=self.hparams.learning_rate)

def colate_fn(batch):
	pos = [s for b in batch for s in b[0]]
	neg = [s for b in batch for s in b[1]]

	return pos, neg

@click.command()
@click.option('--ds_root', type=click.Path(exists=True),
			  default='/home/carmelo/DATASETS/AIC21_Track5_NL_Retrieval')
def main(ds_root):
	pl.seed_everything(1234)

	train = NL_Dataset(ds_root, mode="train")
	val = NL_Dataset(ds_root, mode="val")

	# dataloader
	train_loader = DataLoader(train, batch_size=48, num_workers=8, shuffle=True)
	val_loader = DataLoader(val, batch_size=8, num_workers=4, shuffle=False)

	# model
	model = LiteBert(learning_rate=1e-4)

	# train
	trainer = pl.Trainer(max_epochs=4, gpus=1, accelerator='ddp', accumulate_grad_batches=4)
	trainer.fit(model, train_loader, val_loader)

	if trainer.global_rank == 0:
		# Save pretrained dict
		model.bert_model.save_pretrained('bert_ft_experimental')

if __name__ == '__main__':
	main()
