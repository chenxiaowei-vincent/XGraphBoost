"""Computes the overlap of molecules between two datasets."""

import csv
import os
import sys

from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data.utils import get_data


class Args(Tap):
    data_path_1: str  # Path to first data CSV file
    data_path_2: str  # Path to second data CSV file
    smiles_column_1: str = None  # Name of the column containing SMILES strings for the first data. By default, uses the first column.
    smiles_column_2: str = None  # Name of the column containing SMILES strings for the second data. By default, uses the first column.
    save_intersection_path: str = None  # Path to save intersection at; labeled with data_path 1 header
    save_difference_path: str = None  # Path to save molecules in dataset 1 that are not in dataset 2; labeled with data_path 1 header


def overlap(args: Args):
    data_1 = get_data(path=args.data_path_1, smiles_column=args.smiles_column_1)
    data_2 = get_data(path=args.data_path_2, smiles_column=args.smiles_column_2)

    smiles1 = set(data_1.smiles())
    smiles2 = set(data_2.smiles())
    size_1, size_2 = len(smiles1), len(smiles2)
    intersection = smiles1.intersection(smiles2)
    size_intersect = len(intersection)
    print(f'Size of dataset 1: {size_1}')
    print(f'Size of dataset 2: {size_2}')
    print(f'Size of intersection: {size_intersect}')
    print(f'Size of intersection as frac of dataset 1: {size_intersect / size_1}')
    print(f'Size of intersection as frac of dataset 2: {size_intersect / size_2}')

    if args.save_intersection_path is not None:
        with open(args.data_path_1, 'r') as rf, open(args.save_intersection_path, 'w') as wf:
            reader, writer = csv.reader(rf), csv.writer(wf)
            header = next(reader)
            writer.writerow(header)
            for line in reader:
                if line[0] in intersection:
                    writer.writerow(line)

    if args.save_difference_path is not None:
        with open(args.data_path_1, 'r') as rf, open(args.save_difference_path, 'w') as wf:
            reader, writer = csv.reader(rf), csv.writer(wf)
            header = next(reader)
            writer.writerow(header)
            for line in reader():
                if line[0] not in intersection:
                    writer.writerow(line)


if __name__ == '__main__':
    overlap(Args().parse_args())
