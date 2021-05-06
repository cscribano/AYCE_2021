# -*- coding: utf-8 -*-
# ---------------------

import math
from datetime import datetime
from shutil import copyfile
from time import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from evaluation import inference_on_test
from conf import Conf
from dataset import Extended_TestValDataset
from models import parse_model, parse_optimizer, ExtendedTestable
from utils import reduce_tensor


class Trainer(object):

	def __init__(self, cnf, rank):
		# type: (Conf, int) -> Trainer

		self.cnf = cnf
		self.rank = rank
		self.epoch = 0
		self.log_path = cnf.exp_log_path

		# init model and optimizer
		model, model_args = parse_model(cnf, rank)
		self.model = model(self.cnf, **model_args).to(cnf.device)
		self.load_model_ck()

		# Retrieve dataset class and arguments
		trainset, trainset_args, collate_fn = self.model.train_set
		self.train_all = cnf.data_opts.get('TRAIN_ALL', False)
		if self.train_all:
			trainset_args["mode"] = "train_all"

		valset, valset_args = self.model.val_set

		# initialize DPP
		self.model = DistributedDataParallel(self.model, device_ids=[cnf.gpu_id],
											 output_device=cnf.gpu_id,
											 broadcast_buffers=cnf.world_size > 1)

		# init optimizer
		self.optimizer = parse_optimizer(cnf, self.model)

		# gradient clipping
		self.clip_value = cnf.train_opts.get('CLIP_VALUE', None)

		# init train loader
		self.training_set = trainset(cnf, **trainset_args)
		train_sampler = DistributedSampler(self.training_set, shuffle=True)

		self.train_loader = DataLoader(
			dataset=self.training_set, batch_size=cnf.batch_size, num_workers=cnf.n_workers,
			sampler=train_sampler, pin_memory=False, collate_fn=collate_fn
		)
		self.train_len = len(self.train_loader)

		# init validation loader
		val_set = valset(cnf, **valset_args)
		val_sampler = DistributedSampler(val_set, shuffle=False)

		self.val_loader = DataLoader(
			dataset=val_set, batch_size=4, num_workers=cnf.n_workers,
			pin_memory=False, sampler=val_sampler, collate_fn=collate_fn
		)

		self.val_losses = []
		self.grad_norms = []

		if self.rank == 0:
			# init logging stuffs
			print(f'tensorboard --logdir={cnf.project_log_path.abspath()}\n')
			self.sw = SummaryWriter(self.log_path)
			self.train_losses = []

			# starting values values
			self.best_val_loss = None

		# possibly load checkpoint
		self.load_ck()


	def load_ck(self):
		"""
		load training checkpoint
		"""
		ck_path = self.log_path/'training.ck'
		if ck_path.exists():
			ck = torch.load(ck_path, map_location='cpu')
			print(f'[loading checkpoint \'{ck_path}\']')
			self.epoch = ck['epoch']

			# This is to allow manually changing lr
			for i, g in enumerate(ck['optimizer']['param_groups']):
				g['lr'] = self.optimizer.param_groups[i]['lr']
			self.optimizer.load_state_dict(ck['optimizer'])

			if self.rank == 0:
				self.best_val_loss = ck["best_val_loss"]
			del ck


	def load_model_ck(self, ck_path=None, model_only=False):
		"""
		load Model's state dict
		"""
		if ck_path is None:
			ck_path = self.log_path/'training.ck'

		if ck_path.exists():
			ck = torch.load(ck_path, map_location='cpu')
			if model_only:
				restore_kv = {key.replace("module.", ""): ck[key] for key in ck.keys()}
			else:
				restore_kv = {key.replace("module.", ""): ck["model"][key] for key in ck["model"].keys()}
			self.model.load_state_dict(restore_kv, strict=False)
			del ck

	def save_ck(self):
		"""
		save training checkpoint
		"""
		ck_path = self.log_path/'training.ck'
		if self.rank == 0:

			if ck_path.exists():
				# Avoid losing ck if train get killed during saving
				copyfile(ck_path, self.log_path/'training.ck.prev')

			ck = {
				'epoch': self.epoch,
				'model': self.model.state_dict(),  # TODO: save model not model.module
				'optimizer': self.optimizer.state_dict(),
				'best_val_loss': self.best_val_loss
			}
			torch.save(ck, self.log_path/'training.ck')

			if self.epoch % 100 == 0:
				torch.save(ck, self.log_path / f'training_{self.epoch}.ck')

	def train_step(self, sample, raise_oom=False):
		"""
		Run a single training step
		"""

		try:
			self.optimizer.zero_grad()

			loss = self.model.module.train_loss(sample)

			loss.backward()
			grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).cpu()for p in
												self.model.parameters() if p.grad is not None]), 2)

			# Apply gradient clipping
			self.grad_norms.append(grad_norm)
			if self.clip_value is not None and self.epoch > 10:
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_value)

			self.optimizer.step()

		except RuntimeError as e:
			if 'out of memory' in str(e) and not raise_oom:
				print('| WARNING: ran out of memory, retrying batch')
				for p in self.model.parameters():
					if p.grad is not None:
						del p.grad  # free some memory
				torch.cuda.empty_cache()
				return self.train_step(sample, raise_oom=True)
			else:
				raise e


		return loss.item()

	def train(self):
		"""
		train model for one epoch on the Training-Set.
		"""
		start_time = time()
		self.model.train()

		times = []
		t = time()

		for step, sample in enumerate(self.train_loader):

			l = self.train_step(sample)

			if self.rank == 0:
				self.train_losses.append(l)

				# Display progress
				progress = (step + 1)/self.train_len
				progress_bar = ('█'*int(50*progress)) + ('┈'*(50 - int(50*progress)))
				times.append(time() - t)
				if self.cnf.log_each_step or (not self.cnf.log_each_step and progress == 1):
					print('\r[{}] Epoch {:0{e}d}.{:0{s}d}: │{}│ {:6.2f}% │ Loss: {:.6f} │ Grad: {:.6f} │ ↯: {:5.2f} step/s'.format(
						datetime.now().strftime("%m-%d@%H:%M"), self.epoch, step + 1,
						progress_bar, 100*progress,
						np.mean(self.train_losses), np.mean(self.grad_norms), 1/np.mean(times[-100:]),
						e=math.ceil(math.log10(self.cnf.epochs)),
						s=math.ceil(math.log10(self.train_len)),
						), end='')
				t = time()


		if self.rank == 0:

			# log average loss of this epoch
			mean_epoch_loss = np.mean(self.train_losses)  # type: float
			self.sw.add_scalar(tag='train_loss', scalar_value=mean_epoch_loss, global_step=self.epoch)
			self.train_losses = []
			self.grad_norms = []

			# log epoch duration
			print(f' │ T: {time()-start_time:.2f} s')


	def val_step(self, sample, raise_oom=False):
		"""
		Run a single validation step
		"""

		try:
			val_loss = self.model.module.val_loss(sample)
			# reduce
			val_loss = reduce_tensor(val_loss, self.cnf.world_size)

		except RuntimeError as e:
			if 'out of memory' in str(e) and not raise_oom:
				print('| WARNING: ran out of memory, retrying batch')
				for p in self.model.parameters():
					if p.grad is not None:
						del p.grad  # free some memory
				torch.cuda.empty_cache()
				return self.val_step(sample, raise_oom=True)
			else:
				raise e

		return val_loss.item()

	def validate(self):
		"""
		Validate model on the Validation-Set
		"""

		self.model.eval()

		t = time()
		with torch.no_grad():
			for step, sample in enumerate(self.val_loader):

				val_loss = self.val_step(sample)
				self.val_losses.append(val_loss)

		if self.rank == 0:
			# log average loss on validation set
			mean_val_loss = np.mean(self.val_losses)  # type: float
			self.val_losses = []
			print(f'\t● AVG Loss on VAL-set: {mean_val_loss:.6f} │ T: {time()-t:.2f} s')
			self.sw.add_scalar(tag='val_loss', scalar_value=mean_val_loss, global_step=self.epoch)

			# save best model
			if self.best_val_loss is None or mean_val_loss < self.best_val_loss:
				self.best_val_loss = mean_val_loss
				torch.save(self.model.state_dict(), self.log_path/'best.pth')

	def test(self, modes=("val", ), load_best=False):

		print(">> Started test")
		self.model = self.model.module

		if load_best:
			best_pth = self.log_path/'best.pth'
			if best_pth.exists():
				self.load_model_ck(ck_path=best_pth, model_only=True)
				print("[WARNING]: Loaded model's best checkpoint")
			else:
				print("[WARNING]: Best checkpoint does not exists, running on train checkpoint..")

		# Produce test output of the validation set
		val_model = ExtendedTestable(self.cnf, backbone=self.model)

		for mode in modes:
			# TODO: put this in conf file or somewhere similar
			val_dataset = Extended_TestValDataset(self.cnf, mode)
			inference_on_test(self.cnf, val_dataset, val_model, mode, rank=self.rank)

	def update_lr(self):
		"""
		Update learning rate depending on the epoch
		"""

		lr_dict = self.cnf.train_opts.get("LR_STEPS", {})
		lr_epochs = list(lr_dict.keys())

		lr = self.optimizer.param_groups[0]["lr"]
		if len(lr_epochs) > 0:
			assert len(self.optimizer.param_groups) == 1, "Lr dicts not supported with multiple param groups!"

			for e in lr_epochs:
				if self.epoch >= e:
					lr = lr_dict[e]

		current_lr = self.optimizer.param_groups[0]["lr"]
		if lr != current_lr:
			# Update
			print(f"[TRAINER]: Updating global LR from {current_lr} to {lr}")
			self.optimizer.param_groups[0]['lr'] = lr

	def run(self):
		"""
		start model training procedure (train > validation > checkpoint > repeat)
		"""
		for _ in range(self.epoch, self.cnf.epochs):

			# adjust learning rate
			self.update_lr()

			# Single training epoch
			self.train()

			# if not self.train_all and (self.epoch % self.cnf.val_epoch_step == 0):
			if self.epoch % self.cnf.val_epoch_step == 0:
				self.validate()

			if self.epoch % self.cnf.ck_epoch_step == 0:
				self.save_ck()

			self.epoch += 1

		print(">> Train completed, computing validation result...")
		self.test()
