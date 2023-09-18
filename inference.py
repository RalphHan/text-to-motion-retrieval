import os
from pathlib import Path
import textwrap
import hydra
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from collections import defaultdict
import torch
import logging

import argparse

from models.model import MatchingModel

import evaluation
from utils.visualization import plot_3d_motion

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('run', help='Path to run dir')
    parser.add_argument('--set', default='val', type=str,
                    help='Set on which inference is performed.')
    parser.add_argument('--force', action='store_true', help='Force the evaluation even if the output csv file already exists')
    parser.add_argument('--best_on_metric', default='all', help='select snapshot that optimizes this metric')
    # parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')

    args = parser.parse_args()

    run_path = Path(args.run)
    if not run_path.exists():
        log.warning(f'This path ({run_path}) does not exists. Exiting evaluation.')
        exit(1)

    hydra_cfg = OmegaConf.load(run_path / '.hydra' / 'hydra.yaml')['hydra']
    OmegaConf.register_new_resolver("hydra", lambda x: OmegaConf.select(hydra_cfg, x))

    cfg = OmegaConf.load(run_path / '.hydra' / 'config.yaml')
    print(OmegaConf.to_yaml(cfg))
        
    csv_path = run_path / f'metrics_{args.set}.csv'
    # if csv_path.exists():
    #     log.info(f"The output csv file {csv_path} already exists. Skipping...")
    #     exit(0)

    batch_size = cfg.optim.batch_size

    if args.set == 'val':
        dataset_cfg = cfg.data.val
    elif args.set == 'test':
        dataset_cfg = cfg.data.test
    elif args.set == 'train':
        dataset_cfg = cfg.data.train
    else:
        raise ValueError('Set {} not known!'.format(args.set))

    # load datasets and create dataloaders
    dataloader = hydra.utils.call(dataset_cfg, batch_size=batch_size)

    # construct the model
    model = MatchingModel(cfg)
    if torch.cuda.is_available():
        model.cuda()

    # resume from a saved checkpoint
    ckpts=sorted([x for x in os.listdir(run_path) if x.startswith('ckpt_')],key=lambda x: int(x[6:].split('.')[0]),reverse=True)
    all_metric=defaultdict(list)
    for ckpt in ckpts:
        ckpt_path = run_path / ckpt
        log.info(f"[CKPT]: Loading {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        # perform inference
        motion_feats, caption_feats, motion_labels = evaluation.encode_data(model, dataloader)
        metrics, idxs, postproc_descriptions, ranks, ordered_relevances = evaluation.compute_recall(args.set, caption_feats, motion_feats, motion_labels, dataloader.dataset, return_all=True, top_k_to_return=16)
        metrics['ckpt'] = ckpt
        log.info(metrics)
        for k, v in metrics.items():
            all_metric[k].append(v)
        # save on CSV
        df = pd.DataFrame.from_dict(all_metric)
        df.to_csv(csv_path, index=False)

if __name__ == '__main__':
    main()