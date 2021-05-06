# -*- coding: utf-8 -*-
# ---------------------

import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import signal
import logging
from conf import Conf

import time
import click
import torch.backends.cudnn as cudnn

from trainer import Trainer

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

@click.command()
@click.option('--exp_name', type=str, default=None)
@click.option('--mode', type=click.Choice(['run', 'test', 'val', 'test_both'],
										  case_sensitive=False), default="run")
@click.option('--test_ck', type=click.Choice(['last', 'best'], case_sensitive=False), default="last")
@click.option('--conf_file_path', type=str, default=None)
@click.option('--seed', type=int, default=None)
def main(exp_name, mode, test_ck, conf_file_path, seed):
	# type: (str, str, str, str, int) -> None

	assert torch.backends.cudnn.enabled, "Running without cuDNN is discouraged"

	# if `exp_name` is None,
	# ask the user to enter it
	if exp_name is None:
		exp_name = input('>> experiment name: ')

	# if `exp_name` contains '!',
	# `log_each_step` becomes `False`
	log_each_step = True
	if '!' in exp_name:
		exp_name = exp_name.replace('!', '')
		log_each_step = False

	# if `exp_name` contains a '@' character,
	# the number following '@' is considered as
	# the desired random seed for the experiment
	split = exp_name.split('@')
	if len(split) == 2:
		seed = int(split[1])
		exp_name = split[0]

	cnf = Conf(conf_file_path=conf_file_path, seed=seed, exp_name=exp_name, log=log_each_step)

	print(f'\n▶ Starting Experiment \'{exp_name}\' [seed: {cnf.seed}]')

	# Setup logging
	logging.basicConfig(
		format='[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
		level=logging.INFO,
	)

	print(f'\n▶ Starting Experiment \'{exp_name}\' [seed: {cnf.seed}]')

	cnf_attrs = vars(cnf)
	for k in cnf_attrs:
		s = f'{k} : {cnf_attrs[k]}'
		logging.info(s)

	# Assuming 1 process == 1 GPU
	if not cnf.slurm:
		mp.spawn(
			DDP_Trainer, args=(cnf, mode, test_ck),
			nprocs=cnf.world_size, join=True
		)
	else:
		rank = int(os.environ["SLURM_PROCID"])
		DDP_Trainer(rank, cnf, mode, test_ck)

	signal.signal(signal.SIGINT, cleanup)
	signal.signal(signal.SIGTERM, cleanup)

def init_process(rank, size, host, port, backend='nccl'):
	""" Initialize the distributed environment. """
	os.environ['MASTER_ADDR'] = host
	os.environ['MASTER_PORT'] = str(port)

	dist.init_process_group(backend, rank=rank, world_size=size)

def init_process_slurm(rank, size, jobid, backend='nccl'):
	# type: (int, int, int, str) -> None

	hostfile = f"dist_url.{jobid}.txt"

	if rank == 0:
		dist_url = "tcp://{}:{}".format(Conf.HOSTNAME, Conf.PORT)
		with open(hostfile, "w") as f:
			f.write(dist_url)
	else:
		while not os.path.exists(hostfile):
			time.sleep(1)
		with open(hostfile, "r") as f:
			dist_url = f.read()

	print(f"{dist_url}")

	dist.init_process_group(backend, init_method=dist_url,
								rank=rank, world_size=size)

def cleanup():
	dist.destroy_process_group()

def DDP_Trainer(rank, cnf, mode, test_ck):
	# type: (int, Conf, str, str) -> None

	if cnf.slurm:
		init_process_slurm(rank, cnf.world_size, cnf.jobid)
	else:
		init_process(rank, cnf.world_size, cnf.hostname, cnf.port)

	cnf.setup_device_id(rank)

	print(
		f"Rank {rank + 1}/{cnf.world_size} process initialized.\n"
	)

	trainer = Trainer(cnf, rank)
	cnf.setup_device_id(rank)

	if mode == "run":
		trainer.run()
	elif mode == "test_both":
		trainer.test(modes=("test", "val"), load_best=test_ck == "best")
	else:
		trainer.test(modes=(mode, ), load_best=test_ck == "best")

if __name__ == '__main__':
	main()
