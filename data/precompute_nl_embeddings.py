# -*- coding: utf-8 -*-
# ---------------------

import random
import json
import os
from tqdm import tqdm
import click

import torch
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer, BertModel

# fixed seed for reproducibility
torch.manual_seed(1234)
random.seed(1234)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@click.command()
@click.option('--ds_root', type=click.Path(exists=True),
			  default='/home/carmelo/DATASETS/AIC21_Track5_NL_Retrieval')
@click.option('--mode', type=click.Choice(['train', 'test'], case_sensitive=False), default='test')
def compute_bert_embs(ds_root, mode='train'):
	# type: (str, str) -> None
	"""
	Compute and save the embeddings obtained by BERT, for each sequence a (3,768) tensor
	is produced since 3 different descriptions are provided.

	:param ds_root: Cityflow-NL dataset root (absolute path)
	:param mode: "train" or "test", dataset split to process.
	:return: None
	"""

	# Load BERT model
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
	bert_model = BertModel.from_pretrained('../nlp/bert_ft_experimental').cuda()

	bert_model.eval()

	# Load train json
	if mode == 'train':
		tracks_root = os.path.join(ds_root, 'data/train-tracks.json')
	elif mode == 'test':
		tracks_root = os.path.join(ds_root, 'data/test-queries.json')
	else:
		raise Exception(f"Only train and test are valid split/modes")

	with open(tracks_root, "r") as f:
		tracks = json.load(f)

	keys = list(tracks.keys())

	# Output: {"track_uuid": NL embeddings (3x769) ... }
	output = {}

	for id in tqdm(keys):

		nl = tracks[id]["nl"] if mode == "train" else tracks[id]  # tuple of 3

		with torch.no_grad():
			tokens = tokenizer.batch_encode_plus(nl, padding='longest',
													  return_tensors='pt')
			bert_out = bert_model(tokens['input_ids'].cuda(),
									   attention_mask=tokens['attention_mask'].cuda()).last_hidden_state

		# (3, K, 768) -> (3,768)
		lang_embeds = torch.mean(bert_out, dim=1)
		output[id] = lang_embeds.clone().cpu()

	out_dir = os.path.join(ds_root, f"precomp_nl_{mode}.pt")
	torch.save(output, out_dir)

	# compute statistics
	values = torch.stack(list(output.values()))
	intra_sim = torch.stack([torch.cosine_similarity(v, v.unsqueeze(1), dim=-1).mean() for v in values]).mean()
	inter_sim = torch.cosine_similarity(values, values.unsqueeze(1), dim=-1).mean()
	print(intra_sim, inter_sim, (intra_sim - inter_sim))

if __name__ == '__main__':
	compute_bert_embs()
