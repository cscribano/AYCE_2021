# -*- coding: utf-8 -*-
# ---------------------

import torch
from conf import Conf
from models import *

def parse_model(conf, rank=None):
	# type: (Conf, int) -> tuple

	if conf.model_str == "Extended_LT":
		model = (ExtendedLiteModel, {})
	elif conf.model_str == "Extended_LTNL":
		model = (ExtendedLiteNLModel, {})
	else:
		raise NotImplementedError(f"Model {conf.model_str} is not supported or valid")

	return model

def parse_optimizer(conf, model):
	# type: (Conf, torch.nn.Module) -> torch.optim

	decay = conf.train_opts.get('DECAY', 0.0)  # type:float
	lr_dict = conf.train_opts.get('LEARNING_RATES', {})

	param_dicts = [
		{"params": [p for n, p in model.named_parameters() if not
					any(k in n.split('.') for k in list(lr_dict.keys())) and p.requires_grad]},
	]

	for k in list(lr_dict.keys()):
		p = {
			"params": [p for n, p in model.named_parameters() if k in n.split('.') and p.requires_grad],
			"lr": lr_dict[k],
		}
		param_dicts.append(p)

	assert sum([len(p["params"]) for p in param_dicts]) == len([p for p in model.parameters()])

	if conf.optim_str == "Adam":

		optimizer = torch.optim.Adam(param_dicts, lr=conf.lr, weight_decay=decay)

	elif conf.optim_str == "AdamW":
		optimizer = torch.optim.AdamW(param_dicts, lr=conf.lr, weight_decay=decay)

	elif conf.optim_str == "SGD":
		momentum = conf.train_opts.get('MOMENTUM', 0.9)
		optimizer = torch.optim.SGD(param_dicts, lr=conf.lr, weight_decay=decay, momentum=momentum)

	elif conf.optim_str == "RMSProp":
		optimizer = torch.optim.RMSprop(param_dicts, lr=conf.lr, weight_decay=decay)
	else:
		raise NotImplementedError(f"Optimizer {conf.optim_str} is not supported or valid,"
								  f"you con configure this option in conf.py@parse_optimizer()")

	return optimizer
