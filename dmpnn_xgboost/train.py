"""Trains a model on a dataset."""

from chemprop.args import TrainArgs
from chemprop.train import cross_validate
from chemprop.utils import create_logger
import os

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = TrainArgs().parse_args()
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    model = cross_validate(args, logger)


