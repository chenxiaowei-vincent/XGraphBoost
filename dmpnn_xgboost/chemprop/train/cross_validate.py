from logging import Logger
import os
from typing import Tuple
import pandas as pd
import numpy as np
import torch
import joblib

from .run_training import run_training,get_xgboost_feature,predict_feature
from chemprop.args import TrainArgs
from chemprop.data.utils import get_task_names
from chemprop.utils import makedirs
from morgan_feature import get_morgan_feature,xgboost_cv,xgb_cv_more,xgb_regre_cv,xgb_regre_more


def cross_validate(args: TrainArgs, logger: Logger = None) -> Tuple[float, float]:
    """k-fold cross validation"""
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir

    # Run training on different random seeds for each fold
    dmpnn_scores = []
    scores_df = pd.DataFrame()
    for fold_num in range(args.num_folds):
        if args.dataset_type == 'classification':
            args.data_path = 'molnet_benchmark/molnet_random_'+args.protein+'_c/seed'+str(fold_num+1)+'/train.csv'
            args.separate_test_path = 'molnet_benchmark/molnet_random_'+args.protein+'_c/seed'+str(fold_num+1)+'/val.csv'
            args.separate_val_path = 'molnet_benchmark/molnet_random_'+args.protein+'_c/seed'+str(fold_num+1)+'/test.csv'
        elif args.dataset_type == 'regression':
            args.data_path = 'molnet_benchmark/molnet_random_'+args.protein+'_r/seed'+str(fold_num+1)+'/train.csv'
            args.separate_test_path = 'molnet_benchmark/molnet_random_'+args.protein+'_r/seed'+str(fold_num+1)+'/val.csv'
            args.separate_val_path = 'molnet_benchmark/molnet_random_'+args.protein+'_r/seed'+str(fold_num+1)+'/test.csv'
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        model_scores,model,scaler,df = run_training(args, logger)
        if args.loss_save:
            df.to_csv('/home/cxw/python——work/paper_gcn/dmpnn_epoch_loss/'+args.protein+'loss.csv',index=None)
            # df.to_csv(args.protein+'loss.csv',index=None)
            break
        dmpnn_scores.append(model_scores)
        train_target, train_feature, val_target, val_feature, test_target, test_feature,train_smiles,val_smiles,test_smiles,test_preds = get_xgboost_feature(args, logger,model)
        train_target = pd.DataFrame(train_target)
        train_feature = pd.DataFrame(train_feature)
        val_target = pd.DataFrame(val_target)
        val_feature = pd.DataFrame(val_feature)
        test_target = pd.DataFrame(test_target)
        test_feature = pd.DataFrame(test_feature)
        train_morgan_feature = get_morgan_feature(train_smiles)
        val_morgan_feature = get_morgan_feature(val_smiles)
        test_morgan_feature = get_morgan_feature(test_smiles)
        max_depth_numbers = [2,4,6,8,10]
        learning_rate_numbers = [0.01,0.05,0.1,0.15,0.2]
        min_child_weight_numbers = [2,4,6,8,10]
        if args.dataset_type == 'classification':
            if test_target.shape[1]==1:
                scores = xgboost_cv(max_depth_numbers,learning_rate_numbers,min_child_weight_numbers,
                                       train_feature, train_target,val_feature, val_target,test_feature,test_target,
                                       train_morgan_feature,val_morgan_feature,test_morgan_feature,test_preds)
            else:
                scores = xgb_cv_more(max_depth_numbers,learning_rate_numbers,min_child_weight_numbers,
                                       train_feature, train_target,val_feature, val_target,test_feature,test_target,
                                       train_morgan_feature,val_morgan_feature,test_morgan_feature,test_preds)
            scores.columns = ['type','max_depth','learning_rate','min_child_weight','auc','sn','sp','acc']
            scores_df = pd.concat([scores_df,scores])
        elif args.dataset_type == 'regression':
            if test_target.shape[1]==1:
                scores = xgb_regre_cv(max_depth_numbers, learning_rate_numbers, min_child_weight_numbers,
                                         train_feature, train_target, val_feature, val_target, test_feature, test_target,
                                         train_morgan_feature, val_morgan_feature, test_morgan_feature, test_preds, scaler)
            else:
                scores = xgb_regre_more(max_depth_numbers, learning_rate_numbers, min_child_weight_numbers,
                                         train_feature, train_target, val_feature, val_target, test_feature, test_target,
                                         train_morgan_feature, val_morgan_feature, test_morgan_feature, test_preds, scaler)
            scores.columns = ['type', 'max_depth', 'learning_rate', 'min_child_weight', 'MAE', 'RMSE']
            scores_df = pd.concat([scores_df,scores])

    df_groupby = scores_df.groupby(['type', 'max_depth', 'learning_rate', 'min_child_weight']).mean()
    df_groupby.to_csv(args.protein+'_scores.csv')

    return model




