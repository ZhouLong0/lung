import os
import random
import argparse
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
from src.fewshot.utils import load_cfg_from_cfg_file, merge_cfg_from_list, Logger, get_log_file
from src.fewshot.eval import Evaluator
from src.fewshot.models.ResNet import resnet18
import logging
torch.cuda.empty_cache()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Main')
    cfg = load_cfg_from_cfg_file('config/main_config.yaml')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    model_config = 'config/model_config/model_config.yaml'.format(cfg.dataset, cfg.arch)
    method_config = 'config/methods_config/{}.yaml'.format(cfg.method)
    cfg.update(load_cfg_from_cfg_file(model_config))
    cfg.update(load_cfg_from_cfg_file(method_config))
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg


def main():
    # Configure logging
    
    args = parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    # device = "cpu"
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
    torch.cuda.set_device(0)

    # init logger
    log_file = get_log_file(log_path=args.log_path, dataset=args.dataset,
                            backbone=args.arch, method=args.name_method, sampling=args.sampling)
    logger = Logger(__name__, log_file)

    # create model
    logger.info("=> Creating model '{}'".format(args.arch))
    model = resnet18()
    logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    
    evaluator = Evaluator(device=device, args=args, log_file=log_file)
    evaluator.run_full_evaluation(model=model)
    #return results

if __name__ == "__main__":
    main()
