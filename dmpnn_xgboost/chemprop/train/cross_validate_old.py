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
import xgboost as xgb
from sklearn.metrics import mean_squared_error,median_absolute_error
from sklearn.metrics import roc_curve, auc,precision_recall_curve,confusion_matrix,accuracy_score
from morgan_feature import get_morgan_feature


def cross_validate(args: TrainArgs, logger: Logger = None) -> Tuple[float, float]:
    """k-fold cross validation"""
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    task_names = args.target_columns or get_task_names(args.data_path)

    # Run training on different random seeds for each fold
    dmpnn_scores = []
    dmpnn_xgb_scores = []
    morgan_scores = []
    dmpnn_morgan_scores = []
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
        model_scores,model,scaler = run_training(args, logger)
        dmpnn_scores.append(model_scores)
        train_target, train_feature, val_target, val_feature, test_target, test_feature,train_smiles,val_smiles,test_smiles = get_xgboost_feature(args, logger,model)
        train_target = pd.DataFrame(train_target)
        train_feature = pd.DataFrame(train_feature)
        val_target = pd.DataFrame(val_target)
        val_feature = pd.DataFrame(val_feature)
        test_target = pd.DataFrame(test_target)
        test_feature = pd.DataFrame(test_feature)
        train_morgan_feature = get_morgan_feature(train_smiles)
        val_morgan_feature = get_morgan_feature(val_smiles)
        test_morgan_feature = get_morgan_feature(test_smiles)
        if args.dataset_type == 'classification':
            if test_target.shape[1]==1:
                xgb_gbc = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                            colsample_bytree=1, gamma=1, learning_rate=0.1, max_delta_step=0,
                                            max_depth=4, min_child_weight=8, missing=None, n_estimators=2000,
                                            n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
                                            reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                                            silent=True, subsample=0.8, tree_method='gpu_hist', n_gpus=-1)
                xgb_gbc.fit(train_feature, train_target, eval_set=[(val_feature, val_target)], eval_metric='auc',
                            early_stopping_rounds=200)
                pre_pro = xgb_gbc.predict_proba(test_feature)[:, 1]
                fpr, tpr, threshold = roc_curve(test_target, pre_pro)
                AUC = auc(fpr, tpr)
                pre_pro = [1 if i>0.5 else 0 for i in pre_pro]
                tn, fp, fn, tp = confusion_matrix(test_target, pre_pro).ravel()
                # Sn = TP /（TP + FN）  Sp = TN / (TN+FP)
                Sn = tp/(tp+fn)
                Sp = tn/(tn+fp)
                acc = accuracy_score(test_target, pre_pro)
                dmpnn_xgb_scores.append([AUC,Sn,Sp,acc])
                joblib.dump(xgb_gbc, 'external_test/dmpnn_xgb.model')
                xgb_gbc = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                            colsample_bytree=1, gamma=1, learning_rate=0.1, max_delta_step=0,
                                            max_depth=4, min_child_weight=8, missing=None, n_estimators=2000,
                                            n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
                                            reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                                            silent=True, subsample=0.8, tree_method='gpu_hist', n_gpus=-1)
                xgb_gbc.fit(train_morgan_feature, train_target, eval_set=[(val_morgan_feature, val_target)], eval_metric='auc',
                            early_stopping_rounds=200)
                pre_pro = xgb_gbc.predict_proba(test_morgan_feature)[:, 1]
                fpr, tpr, threshold = roc_curve(test_target, pre_pro)
                AUC = auc(fpr, tpr)
                pre_pro = [1 if i > 0.5 else 0 for i in pre_pro]
                tn, fp, fn, tp = confusion_matrix(test_target, pre_pro).ravel()
                # Sn = TP /（TP + FN）  Sp = TN / (TN+FP)
                Sn = tp / (tp + fn)
                Sp = tn / (tn + fp)
                acc = accuracy_score(test_target, pre_pro)
                morgan_scores.append([AUC, Sn, Sp, acc])
                joblib.dump(xgb_gbc, 'external_test/morgan_xgb.model')
                train_gcn_mor_feature = pd.concat([train_feature,train_morgan_feature],axis=1)
                val_gcn_mor_feature = pd.concat([val_feature,val_morgan_feature],axis=1)
                test_gcn_mor_feature = pd.concat([test_feature,test_morgan_feature],axis=1)
                train_gcn_mor_feature.columns = val_gcn_mor_feature.columns = test_gcn_mor_feature.columns = range(train_gcn_mor_feature.shape[1])
                xgb_gbc = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                            colsample_bytree=1, gamma=1, learning_rate=0.1, max_delta_step=0,
                                            max_depth=4, min_child_weight=8, missing=None, n_estimators=2000,
                                            n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
                                            reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                                            silent=True, subsample=0.8, tree_method='gpu_hist', n_gpus=-1)
                xgb_gbc.fit(train_gcn_mor_feature, train_target, eval_set=[(val_gcn_mor_feature, val_target)], eval_metric='auc',
                            early_stopping_rounds=200)
                pre_pro = xgb_gbc.predict_proba(test_gcn_mor_feature)[:, 1]
                fpr, tpr, threshold = roc_curve(test_target, pre_pro)
                AUC = auc(fpr, tpr)
                pre_pro = [1 if i > 0.5 else 0 for i in pre_pro]
                tn, fp, fn, tp = confusion_matrix(test_target, pre_pro).ravel()
                Sn = tp / (tp + fn)
                Sp = tn / (tn + fp)
                acc = accuracy_score(test_target, pre_pro)
                dmpnn_morgan_scores.append([AUC, Sn, Sp, acc])
                joblib.dump(xgb_gbc, 'external_test/dmpnn_morgan_xgb.model')


            else:
                aucs=[]
                for i in range(test_target.shape[1]):
                    xgb_gbc = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                                colsample_bytree=1, gamma=1, learning_rate=0.1, max_delta_step=0,
                                                max_depth=4, min_child_weight=8, missing=None, n_estimators=2000,
                                                n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
                                                reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                                                silent=True, subsample=0.8, tree_method='gpu_hist', n_gpus=-1)
                    if max(val_target[i])==0 or max(train_target[i])==0 or max(test_target[i])==0:
                        continue
                    xgb_gbc.fit(train_feature, train_target[i], eval_set=[(val_feature, val_target[i])], eval_metric='auc',
                                early_stopping_rounds=100)
                    pre_pro = xgb_gbc.predict_proba(test_feature)[:, 1]
                    fpr, tpr, threshold = roc_curve(test_target[i], pre_pro)
                    AUC = auc(fpr, tpr)
                    if args.metric == "prc-auc":
                        precision, recall, _ = precision_recall_curve(test_target[i], pre_pro)
                        AUC = auc(recall, precision)
                    pre_pro = [1 if i > 0.5 else 0 for i in pre_pro]
                    tn, fp, fn, tp = confusion_matrix(test_target, pre_pro).ravel()
                    Sn = tp / (tp + fn)
                    Sp = tn / (tn + fp)
                    acc = accuracy_score(test_target, pre_pro)
                    aucs.append([AUC,Sn,Sp,acc])
                dmpnn_xgb_scores.append([np.mean(aucs)])
        elif args.dataset_type == 'regression':
            if test_target.shape[1]==1:
                xgb_gbc = xgb.XGBRegressor(learn_rate=0.1, max_depth=4, min_child_weight=10, gamma=1, subsample=0.8,
                                           colsample_bytree=0.8,
                                           reg_alpha=0.8, objective='reg:linear', n_estimators=2000, tree_method='gpu_hist',
                                           n_gpus=-1)
                xgb_gbc.fit(train_feature, train_target, eval_set=[(val_feature, val_target)], eval_metric='rmse',
                            early_stopping_rounds=200)
                y_pred = xgb_gbc.predict(test_feature)
                y_pred = scaler.inverse_transform(y_pred)
                y_test = test_target.astype('float')
                MSE = mean_squared_error(y_test, y_pred)
                RMSE = MSE ** 0.5
                MAE = median_absolute_error(y_test, y_pred)
                dmpnn_xgb_scores.append([RMSE,MAE])
                joblib.dump(xgb_gbc, 'external_test/dmpnn_xgb.model')
                xgb_gbc = xgb.XGBRegressor(learn_rate=0.1, max_depth=4, min_child_weight=10, gamma=1, subsample=0.8,
                                           colsample_bytree=0.8,
                                           reg_alpha=0.8, objective='reg:linear', n_estimators=2000, tree_method='gpu_hist',
                                           n_gpus=-1)
                xgb_gbc.fit(train_morgan_feature, train_target, eval_set=[(val_morgan_feature, val_target)], eval_metric='rmse',
                            early_stopping_rounds=200)
                y_pred = xgb_gbc.predict(test_morgan_feature)
                y_pred = scaler.inverse_transform(y_pred)
                MSE = mean_squared_error(y_test, y_pred)
                RMSE = MSE ** 0.5
                MAE = median_absolute_error(y_test, y_pred)
                morgan_scores.append([RMSE,MAE])
                joblib.dump(xgb_gbc, 'external_test/morgan_xgb.model')
                train_gcn_mor_feature = pd.concat([train_feature,train_morgan_feature],axis=1)
                val_gcn_mor_feature = pd.concat([val_feature,val_morgan_feature],axis=1)
                test_gcn_mor_feature = pd.concat([test_feature,test_morgan_feature],axis=1)
                train_gcn_mor_feature.columns = val_gcn_mor_feature.columns = test_gcn_mor_feature.columns = range(train_gcn_mor_feature.shape[1])

                xgb_gbc = xgb.XGBRegressor(learn_rate=0.1, max_depth=4, min_child_weight=10, gamma=1, subsample=0.8,
                                           colsample_bytree=0.8,
                                           reg_alpha=0.8, objective='reg:linear', n_estimators=2000, tree_method='gpu_hist',
                                           n_gpus=-1)
                xgb_gbc.fit(train_gcn_mor_feature, train_target, eval_set=[(val_gcn_mor_feature, val_target)], eval_metric='rmse',
                            early_stopping_rounds=200)
                y_pred = xgb_gbc.predict(test_gcn_mor_feature)
                y_pred = scaler.inverse_transform(y_pred)
                MSE = mean_squared_error(y_test, y_pred)
                RMSE = MSE ** 0.5
                MAE = median_absolute_error(y_test, y_pred)
                dmpnn_morgan_scores.append([RMSE,MAE])
                joblib.dump(xgb_gbc, 'external_test/dmpnn_morgan_xgb.model')

            else:
                MAEs = []
                for i in range(test_target.shape[1]):
                    xgb_gbc = xgb.XGBRegressor(learn_rate=0.1, max_depth=4, min_child_weight=10, gamma=1, subsample=0.8,
                                               colsample_bytree=0.8,
                                               reg_alpha=0.8, objective='reg:linear', n_estimators=2000,
                                               tree_method='gpu_hist',
                                               n_gpus=-1)
                    xgb_gbc.fit(train_feature, train_target[i], eval_set=[(val_feature, val_target[i])], eval_metric='rmse',
                                early_stopping_rounds=200)
                    y_pred = xgb_gbc.predict(test_feature)
                    y_test = test_target[i].astype('float')
                    MSE = mean_squared_error(y_test, y_pred)
                    RMSE = MSE ** 0.5
                    MAE = median_absolute_error(y_test, y_pred)
                    MAEs.append([MAE,RMSE])
                dmpnn_xgb_scores.append([np.mean(MAEs)])

    dmpnn_scores = np.array(dmpnn_scores)
    # Report scores across models
    dmpnn_scores = np.nanmean(dmpnn_scores, axis=1)  # average score for each model across tasks
    dmpnn_mean_score, dmpnn_std_score = np.nanmean(dmpnn_scores), np.nanstd(dmpnn_scores)
    print('three dmpnn test = ',dmpnn_scores)
    info(f'Overall dmpnn test {args.metric} = {dmpnn_mean_score:.6f} +/- {dmpnn_std_score:.6f}')

    dmpnn_xgb_scores = np.nanmean(dmpnn_xgb_scores, axis=1)  # average score for each model across tasks
    dmpnn_xgb_mean_score, dmpnn_xgb_std_score = np.nanmean(dmpnn_xgb_scores), np.nanstd(dmpnn_xgb_scores)
    print('three dmpnn_xgb_test = ',dmpnn_xgb_scores)
    info(f'Overall dmpnn_xgb_test {args.metric} = {dmpnn_xgb_mean_score:.6f} +/- {dmpnn_xgb_std_score:.6f}')

    morgan_scores = np.nanmean(morgan_scores, axis=1)  # average score for each model across tasks
    morgan_mean_score, morgan_std_score = np.nanmean(morgan_scores), np.nanstd(morgan_scores)
    print('three morgen_test = ',morgan_scores)
    info(f'Overall morgen_test {args.metric} = {morgan_mean_score:.6f} +/- {morgan_std_score:.6f}')

    dmpnn_morgan_scores = np.nanmean(dmpnn_morgan_scores, axis=1)  # average score for each model across tasks
    dmpnn_morgan_mean_score, dmpnn_morgan_std_score = np.nanmean(dmpnn_morgan_scores), np.nanstd(dmpnn_morgan_scores)
    print('three dmpnn_morgan_scores = ',dmpnn_morgan_scores)
    info(f'Overall dmpnn_morgen_test {args.metric} = {dmpnn_morgan_mean_score:.6f} +/- {dmpnn_morgan_std_score:.6f}')
    return model
    # torch.save(model, 'external_test/model.pth')


def vs_predict(args,model,logger,external_test_path):

    # model = torch.load('external_test/model.pth')  # 直接加载模型
    dmpnn_xgb = joblib.load('external_test/dmpnn_xgb.model')
    morgan_xgb = joblib.load('external_test/morgan_xgb.model')
    dmpnn_morgan_xgb = joblib.load('external_test/dmpnn_morgan_xgb.model')
    external_test_smiles,external_test_feature,external_test_preds,external_test_targets = predict_feature(args, logger, model, external_test_path)
    external_test_feature = pd.DataFrame(external_test_feature)
    external_test_morgan_feature = get_morgan_feature(external_test_smiles)
    external_test_gnn_mor_feature = pd.concat([external_test_feature, external_test_morgan_feature], axis=1)
    external_test_gnn_mor_feature.columns = range(external_test_gnn_mor_feature.shape[1])

    dmpnn_xgb_pre_pro = dmpnn_xgb.predict_proba(external_test_feature)[:,1]
    morgan_xgb_pre_pro = morgan_xgb.predict_proba(external_test_morgan_feature)[:,1]
    dmpnn_morgan_xgb_pre_pro = dmpnn_morgan_xgb.predict_proba(external_test_gnn_mor_feature)[:,1]

    input_data = pd.DataFrame([external_test_smiles, dmpnn_xgb_pre_pro,morgan_xgb_pre_pro,dmpnn_morgan_xgb_pre_pro,external_test_targets]).T
    input_data.columns = ['test_smile', 'dmpnn_xgb_pre_pro', 'morgan_xgb_pre_pro', 'dmpnn_morgan_xgb_pre_pro','target']
    input_data.to_csv('external_test/input_predict.csv',index=None)


