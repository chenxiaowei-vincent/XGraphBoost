from typing import List

import torch
import torch.nn as nn
from tqdm import tqdm

from chemprop.data import MoleculeDataLoader, MoleculeDataset, StandardScaler


def predict(model: nn.Module,
            data_loader: MoleculeDataLoader,
            disable_progress_bar: bool = False,
            scaler: StandardScaler = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data_loader: A MoleculeDataLoader.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    model.eval()

    preds = []
    feature_list = []

    for batch in tqdm(data_loader, disable=disable_progress_bar):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch = batch.batch_graph(), batch.features()

        # Make predictions
        with torch.no_grad():
            batch_preds,feature = model(mol_batch, features_batch)

        batch_preds = batch_preds.data.cpu().numpy()
        feature = feature.data.cpu().numpy()

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)
        feature = feature.tolist()
        feature_list.extend(feature)

    return preds,feature_list
