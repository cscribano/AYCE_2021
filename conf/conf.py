# -*- coding: utf-8 -*-
# ---------------------

import os

PYTHONPATH = '..:.'
if os.environ.get('PYTHONPATH', default=None) is None:
	os.environ['PYTHONPATH'] = PYTHONPATH
else:
	os.environ['PYTHONPATH'] += (':' + PYTHONPATH)

import yaml
import socket
import random
import torch
import numpy as np
from path import Path
from typing import Optional

def set_seed(seed=None):
	# type: (Optional[int]) -> int
	"""
	set the random seed using the required value (`seed`)
	or a random value if `seed` is `None`
	:return: the newly set seed
	"""
	if seed is None:
		seed = random.randint(1, 10000)
	random.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	return seed

def find_free_port():
	s = socket.socket()
	s.bind(('', 0))            # Bind to a free port provided by the host.
	return s.getsockname()[1]  # Return the port number assigned.

class Conf(object):
	# HOSTNAME = socket.gethostname()
	HOSTNAME = socket.gethostname()  # socket.gethostbyname(socket.gethostname())
	PORT = find_free_port()
	OUT_PATH = Path(__file__).parent.parent

	def __init__(self, conf_file_path=None, seed=None, exp_name=None, log=True, device='cuda'):
		# type: (str, int, str, bool, str) -> None
		"""
		:param conf_file_path: optional path of the configuration file
		:param seed: desired seed for the RNG; if `None`, it will be chosen randomly
		:param exp_name: name of the experiment
		:param log: `True` if you want to log each step; `False` otherwise
		:param device: torch device you want to use for train/test
			:example values: 'cpu', 'cuda', 'cuda:5', ...
		"""
		self.exp_name = exp_name
		self.log_each_step = log
		self.device = device

		self.hostname = Conf.HOSTNAME
		self.port = Conf.PORT

		# Check if we are running a slurm job
		self.slurm = os.environ.get("SLURM_TASK_PID") is not None
		if self.slurm:
			print(">> Detected SLURM")
			self.tmpdir = os.environ.get("TMPDIR")
		else:
			self.tmpdir = None

		# DDP STUFF
		self.gpu_id = 0
		if not self.slurm:
			self.world_size = torch.cuda.device_count()
			self.jobid = None
		else:
			self.world_size = int(os.environ["SLURM_NPROCS"])
			assert (self.world_size % torch.cuda.device_count()) == 0, "Use 1 task per GPU!"
			self.jobid = os.environ["SLURM_JOBID"]

		print(f"Training on {self.world_size} GPUs")

		# print project name and host name
		self.project_name = Path(__file__).parent.parent.basename()
		m_str = f'┃ {self.project_name}@{Conf.HOSTNAME} ┃'
		u_str = '┏' + '━' * (len(m_str) - 2) + '┓'
		b_str = '┗' + '━' * (len(m_str) - 2) + '┛'
		print(u_str + '\n' + m_str + '\n' + b_str)

		# project root
		self.project_root = Conf.OUT_PATH


		# set random seed
		self.seed = set_seed(seed)  # type: int

		# if the configuration file is not specified
		# try to load a configuration file based on the experiment name
		tmp = Path(os.path.join(os.path.dirname(__file__), 'experiments', f"{self.exp_name}.yaml"))
		if conf_file_path is None and tmp.exists():
			conf_file_path = tmp

		# read the YAML configuation file
		if conf_file_path is None:
			y = {}
		else:
			conf_file = open(conf_file_path, 'r')
			y = yaml.load(conf_file)

		# read configuration parameters from YAML file
		# or set their default value
		self.base_opts = y.get('BASE_OPTS', {})  # type: dict

		self.epochs = self.base_opts.get('EPOCHS', 10)  # type: int
		self.n_workers = self.base_opts.get('N_WORKERS', 0)  # type: int
		if self.device == 'cuda' and self.base_opts.get('DEVICE', None) is not None:
			self.device = self.base_opts.get('DEVICE')  # type: str
		self.val_epoch_step = self.base_opts.get('VAL_EPOCHS_STEP', 1) # type: int
		self.ck_epoch_step = self.base_opts.get('CK_EPOCHS_STEP', 1) # type: int

		self.model_str = self.base_opts['MODEL']  # type: str

		# Dataset stuff
		self.data_opts = y.get('DATA_OPTS', {})  # type: dict
		self.data_root = self.data_opts.get('DATASET', '.')  # type: Path
		self.batch_size = self.data_opts.get('BATCH_SIZE', 8)  # type: int

		# define output paths
		logdir = self.data_opts.get('LOG_DIR', '') # type: Path
		if logdir != '':
			logdir = Path(logdir)
			self.project_log_path = Path(logdir / 'log' / self.project_name)
		else:
			self.project_log_path = Path(Conf.OUT_PATH / 'log' / self.project_name)

		self.exp_log_path = self.project_log_path / exp_name

		# Training options
		self.train_opts = y.get('TRAIN_OPTS', {})  # type: dict
		self.lr = self.train_opts.get('LR', 0.0001)  # type: float
		self.optim_str = self.train_opts['OPTIMIZER']  # type: str

		# Model hp
		self.model_opts = y.get('MODEL_OPTS', {})  # type: dict
		'''
		DO NOT PUT MODEL-SPECIFIC OPTIONS IN CONF
		LIKE THOSE:
		self.out_dim = y.get('OUT_DIM', 256)  # type: int
		self.positive_tresh = y.get('POSITIVE_TRESH', 0.5)  # type: float
		'''

	@property
	def is_cuda(self):
		# type: () -> bool
		"""
		:return: `True` if the required device is 'cuda'; `False` otherwise
		"""
		return 'cuda' in self.device

	def setup_device_id(self, rank):
		if self.slurm:
			self.gpu_id = rank % torch.cuda.device_count()  # Assuming an equal number of gpus per node
		else:
			self.gpu_id = rank

		if self.device == "cuda":
			self.device = f"cuda:{self.gpu_id}"
