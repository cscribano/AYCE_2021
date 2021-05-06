# -*- coding: utf-8 -*-
# ---------------------

import os
import json
import torch
from torch.utils.data import DataLoader
from path import Path
from tqdm import tqdm

from conf import Conf
from models import TestableModel
from dataset.base_datasets import TestDataset

import torch.nn.functional as F
import torch.distributed as dist


def inference_on_test(cnf, dataset, model, mode="test", rank=0):
    # type: (Conf, TestDataset, TestableModel, str, int) -> None

    """
    Run inference on the test/validation set
    :param dataset: a Dataset object that inherits from TestableModel
    :param model: object of a class that inherit from TestDataset
    :return: None
    """

    model.backbone.eval()

    # distance function
    if hasattr(model.backbone, 'distance'):
        dist_fn = model.backbone.distance()
    else:
        dist_fn = F.pairwise_distance

    # Create dataloader object
    dataloader = DataLoader(
        dataset=dataset, batch_size=1, num_workers=8, shuffle=False
    )

    # retrieve queries
    num_queries = dataset.num_queries()
    assert num_queries == len(dataloader)

    # Loop over all queries
    results = {}
    seq_embeddings = {}
    nl_embeddings = {}

    """
    Result dict in the format {"nlp-uuid": [<track-uuid-1>,...<track-uuid-n>]
    """
    for index, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
        if (index % cnf.world_size) != rank:
            continue

        query_id, q = dataset.get_query(index)  # (3,)

        with torch.no_grad():
            uuid, images, boxes, egocrop = sample

            seq_emb, nl_emb = model.test_step(images, boxes, q, egocrop, precomp=None)
            nl_embeddings[query_id] = nl_emb
            seq_embeddings[uuid[0]] = seq_emb

    if cnf.world_size > 1:
        int_output = Path.abspath(cnf.exp_log_path) / f"{mode}_intermediate_{rank}.pt"
        torch.save([seq_embeddings, nl_embeddings], int_output)

    dist.barrier()

    # Merge..
    if rank == 0:

        if cnf.world_size > 1:
            for i in range(cnf.world_size):
                int_output = Path.abspath(cnf.exp_log_path) / f"{mode}_intermediate_{i}.pt"
                interm = torch.load(int_output)

                seq_embeddings.update(interm[0])
                nl_embeddings.update(interm[1])

                os.remove(int_output)

        # Sanity check
        assert num_queries == len(nl_embeddings.keys())

        # Compute all possible distance pairs
        for q_id in list(nl_embeddings.keys()):

            track_scores = {}
            for s_id in list(seq_embeddings.keys()):
                # TODO: try cpu vs gpu
                d = dist_fn(seq_embeddings[s_id], nl_embeddings[q_id][0])
                d = torch.mean(d, dim=0)
                track_scores[s_id] = d.item()

            # Sorted closer to farther!
            track_scores = sorted(track_scores, key=track_scores.get, reverse=False)  # <--- The closer=the better!!!!
            results[q_id] = list(track_scores)

        # save output
        results_file = Path.abspath(cnf.exp_log_path) / f'{mode}_result.json'

        with open(results_file, "w") as f:
            json.dump(results, f)

