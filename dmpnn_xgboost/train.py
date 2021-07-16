"""Trains a model on a dataset."""
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from chemprop.args import TrainArgs
from chemprop.train import cross_validate,cross_validate_mechine
from chemprop.utils import create_logger


if __name__ == '__main__':
    args = TrainArgs().parse_args()
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    model = cross_validate(args, logger)
    # cross_validate_mechine(args, logger)


