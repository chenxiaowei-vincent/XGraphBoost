from math import isnan
import pandas as pd
from rdkit.Chem import AllChem
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error,median_absolute_error
from sklearn.metrics import roc_curve, auc,precision_recall_curve,confusion_matrix,accuracy_score



def ZW6(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol,3,nBits=256).ToBitString()
def get_mol(smile):
    try:
        N = len(smile)
        nump = []
        for i in range(0, N):
            mol = AllChem.MolFromSmiles(smile[i])
            nump.append(mol)
        return nump
    except:
        nump = []
        mol = AllChem.MolFromSmiles(smile)
        nump.append(mol)
        return nump

def get_morgan_feature(smile):
    mol = get_mol(smile)
    data = []
    for i in range(len(mol)):
        try:
            data.append([smile[i], ZW6(mol[i])])
        except:
            continue
    jak_feature = pd.DataFrame(data, columns=['smiles','ZW6'])
    num_frame6 = []
    for i in range(len(jak_feature['ZW6'])):
        num_frame6.append([x for x in jak_feature['ZW6'][i]])
    jak_zw6 = pd.DataFrame(num_frame6,dtype=np.float)
    return jak_zw6

def xgboost_cv(max_depth_numbers,learning_rate_numbers,min_child_weight_numbers,
               train_feature, train_target,val_feature, val_target,test_feature,test_target,
               train_morgan_feature,val_morgan_feature,test_morgan_feature,test_preds):
    scores = []
    fpr, tpr, threshold = roc_curve(test_target, [i[0] for i in test_preds])
    AUC = auc(fpr, tpr)
    pre_pro = [1 if i > 0.5 else 0 for i in [i[0] for i in test_preds]]
    tn, fp, fn, tp = confusion_matrix(test_target, pre_pro).ravel()
    Sn = tp / (tp + fn)
    Sp = tn / (tn + fp)
    acc = accuracy_score(test_target, pre_pro)
    xgb_type = 'dmpnn'
    scores.append([xgb_type, 'none', 'none', 'none', AUC, Sn, Sp, acc])


    for max_depth_number in max_depth_numbers:
        for learning_rate_number in learning_rate_numbers:
            for min_child_weight_number in min_child_weight_numbers:
                xgb_gbc = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                            colsample_bytree=1, gamma=1, learning_rate=learning_rate_number, max_delta_step=0,
                                            max_depth=max_depth_number, min_child_weight=min_child_weight_number, missing=None, n_estimators=2000,
                                            n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
                                            reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                                            silent=True, subsample=1, tree_method='gpu_hist', n_gpus=-1)
                xgb_gbc.fit(train_feature, train_target, eval_set=[(val_feature, val_target)], eval_metric='auc',
                            early_stopping_rounds=200)
                pre_pro = xgb_gbc.predict_proba(test_feature)[:, 1]
                fpr, tpr, threshold = roc_curve(test_target, pre_pro)
                AUC = auc(fpr, tpr)
                pre_pro = [1 if i > 0.5 else 0 for i in pre_pro]
                tn, fp, fn, tp = confusion_matrix(test_target, pre_pro).ravel()
                Sn = tp / (tp + fn)
                Sp = tn / (tn + fp)
                acc = accuracy_score(test_target, pre_pro)
                xgb_type = 'dmpnn+xgb'
                scores.append([xgb_type,max_depth_number,learning_rate_number,min_child_weight_number,AUC,Sn,Sp,acc])
    for max_depth_number in max_depth_numbers:
        for learning_rate_number in learning_rate_numbers:
            for min_child_weight_number in min_child_weight_numbers:
                xgb_gbc = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                            colsample_bytree=1, gamma=1, learning_rate=learning_rate_number, max_delta_step=0,
                                            max_depth=max_depth_number, min_child_weight=min_child_weight_number, missing=None, n_estimators=2000,
                                            n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
                                            reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                                            silent=True, subsample=1, tree_method='gpu_hist', n_gpus=-1)
                xgb_gbc.fit(train_morgan_feature, train_target, eval_set=[(val_morgan_feature, val_target)],
                            eval_metric='auc',
                            early_stopping_rounds=200)
                pre_pro = xgb_gbc.predict_proba(test_morgan_feature)[:, 1]
                fpr, tpr, threshold = roc_curve(test_target, pre_pro)
                AUC = auc(fpr, tpr)
                pre_pro = [1 if i > 0.5 else 0 for i in pre_pro]
                tn, fp, fn, tp = confusion_matrix(test_target, pre_pro).ravel()
                Sn = tp / (tp + fn)
                Sp = tn / (tn + fp)
                acc = accuracy_score(test_target, pre_pro)
                xgb_type = 'morgan+xgb'
                scores.append([xgb_type,max_depth_number,learning_rate_number,min_child_weight_number,AUC,Sn,Sp,acc])
    for max_depth_number in max_depth_numbers:
        for learning_rate_number in learning_rate_numbers:
            for min_child_weight_number in min_child_weight_numbers:
                train_gcn_mor_feature = pd.concat([train_feature, train_morgan_feature], axis=1)
                val_gcn_mor_feature = pd.concat([val_feature, val_morgan_feature], axis=1)
                test_gcn_mor_feature = pd.concat([test_feature, test_morgan_feature], axis=1)
                train_gcn_mor_feature.columns = val_gcn_mor_feature.columns = test_gcn_mor_feature.columns = range(
                    train_gcn_mor_feature.shape[1])
                xgb_gbc = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                            colsample_bytree=1, gamma=1, learning_rate=learning_rate_number, max_delta_step=0,
                                            max_depth=max_depth_number, min_child_weight=min_child_weight_number, missing=None, n_estimators=2000,
                                            n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
                                            reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                                            silent=True, subsample=1, tree_method='gpu_hist', n_gpus=-1)
                xgb_gbc.fit(train_gcn_mor_feature, train_target, eval_set=[(val_gcn_mor_feature, val_target)],
                            eval_metric='auc',
                            early_stopping_rounds=200)
                pre_pro = xgb_gbc.predict_proba(test_gcn_mor_feature)[:, 1]
                fpr, tpr, threshold = roc_curve(test_target, pre_pro)
                AUC = auc(fpr, tpr)
                pre_pro = [1 if i > 0.5 else 0 for i in pre_pro]
                tn, fp, fn, tp = confusion_matrix(test_target, pre_pro).ravel()
                Sn = tp / (tp + fn)
                Sp = tn / (tn + fp)
                acc = accuracy_score(test_target, pre_pro)
                xgb_type = 'dmpnn+morgan+xgb'
                scores.append([xgb_type,max_depth_number,learning_rate_number,min_child_weight_number,AUC,Sn,Sp,acc])
    scores_df = pd.DataFrame(scores)
    return scores_df

def xgb_cv_more(max_depth_numbers,learning_rate_numbers,min_child_weight_numbers,
               train_feature, train_target,val_feature, val_target,test_feature,test_target,
               train_morgan_feature,val_morgan_feature,test_morgan_feature,test_preds):
    train_gcn_mor_feature = pd.concat([train_feature, train_morgan_feature], axis=1)
    val_gcn_mor_feature = pd.concat([val_feature, val_morgan_feature], axis=1)
    test_gcn_mor_feature = pd.concat([test_feature, test_morgan_feature], axis=1)
    train_gcn_mor_feature.columns = val_gcn_mor_feature.columns = test_gcn_mor_feature.columns = range(
        train_gcn_mor_feature.shape[1])
    scores = []
    dmpnn = []
    dmpnn_xgb = []
    morgan_xgb = []
    dmpnn_morgan_xgb = []
    for k in range(test_target.shape[1]):
        fpr, tpr, threshold = roc_curve(test_target[k], [i[k] for i in test_preds])
        AUC = auc(fpr, tpr)
        if isnan(AUC):
            continue
        else:
            pre_pro = [1 if i > 0.5 else 0 for i in [i[k] for i in test_preds]]
            tn, fp, fn, tp = confusion_matrix(test_target[k], pre_pro).ravel()
            Sn = tp / (tp + fn)
            Sp = tn / (tn + fp)
            acc = accuracy_score(test_target[k], pre_pro)
            dmpnn.append([AUC,Sn,Sp,acc])
    dmpnn = [i for i in np.mean(dmpnn, axis=0)]
    xgb_type = 'dmpnn'
    scores.append([xgb_type, 'none', 'none', 'none', dmpnn[0], dmpnn[1], dmpnn[2], dmpnn[3]])
    for max_depth_number in max_depth_numbers:
        for learning_rate_number in learning_rate_numbers:
            for min_child_weight_number in min_child_weight_numbers:
                for i in range(test_target.shape[1]):
                    xgb_gbc = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                                colsample_bytree=1, gamma=1, learning_rate=learning_rate_number, max_delta_step=0,
                                                max_depth=max_depth_number, min_child_weight=min_child_weight_number, missing=None, n_estimators=2000,
                                                n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
                                                reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                                                silent=True, subsample=0.8, tree_method='gpu_hist', n_gpus=-1)
                    if max(val_target[i]) == 0 or max(train_target[i]) == 0 or max(test_target[i]) == 0:
                        continue
                    xgb_gbc.fit(train_feature, train_target[i], eval_set=[(val_feature, val_target[i])], eval_metric='auc',
                                early_stopping_rounds=100)
                    pre_pro = xgb_gbc.predict_proba(test_feature)[:, 1]
                    fpr, tpr, threshold = roc_curve(test_target[i], pre_pro)
                    AUC = auc(fpr, tpr)
                    pre_pro = [1 if i > 0.5 else 0 for i in pre_pro]
                    tn, fp, fn, tp = confusion_matrix(test_target[i], pre_pro).ravel()
                    Sn = tp / (tp + fn)
                    Sp = tn / (tn + fp)
                    acc = accuracy_score(test_target[i], pre_pro)
                    dmpnn_xgb.append([AUC, Sn, Sp, acc])
                    xgb_gbc = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                                colsample_bytree=1, gamma=1, learning_rate=learning_rate_number,
                                                max_delta_step=0,
                                                max_depth=max_depth_number, min_child_weight=min_child_weight_number,
                                                missing=None, n_estimators=2000,
                                                n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
                                                reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                                                silent=True, subsample=1, tree_method='gpu_hist', n_gpus=-1)
                    xgb_gbc.fit(train_morgan_feature, train_target[i], eval_set=[(val_morgan_feature, val_target[i])],
                                eval_metric='auc',early_stopping_rounds=200)
                    pre_pro = xgb_gbc.predict_proba(test_morgan_feature)[:, 1]
                    fpr, tpr, threshold = roc_curve(test_target[i], pre_pro)
                    AUC = auc(fpr, tpr)
                    pre_pro = [1 if i > 0.5 else 0 for i in pre_pro]
                    tn, fp, fn, tp = confusion_matrix(test_target[i], pre_pro).ravel()
                    Sn = tp / (tp + fn)
                    Sp = tn / (tn + fp)
                    acc = accuracy_score(test_target[i], pre_pro)
                    morgan_xgb.append([AUC, Sn, Sp, acc])
                    xgb_gbc = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                                colsample_bytree=1, gamma=1, learning_rate=learning_rate_number,
                                                max_delta_step=0,
                                                max_depth=max_depth_number, min_child_weight=min_child_weight_number,
                                                missing=None, n_estimators=2000,
                                                n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
                                                reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                                                silent=True, subsample=1, tree_method='gpu_hist', n_gpus=-1)
                    xgb_gbc.fit(train_gcn_mor_feature, train_target[i], eval_set=[(val_gcn_mor_feature, val_target[i])],
                                eval_metric='auc',
                                early_stopping_rounds=200)
                    pre_pro = xgb_gbc.predict_proba(test_gcn_mor_feature)[:, 1]
                    fpr, tpr, threshold = roc_curve(test_target[i], pre_pro)
                    AUC = auc(fpr, tpr)
                    pre_pro = [1 if i > 0.5 else 0 for i in pre_pro]
                    tn, fp, fn, tp = confusion_matrix(test_target[i], pre_pro).ravel()
                    Sn = tp / (tp + fn)
                    Sp = tn / (tn + fp)
                    acc = accuracy_score(test_target[i], pre_pro)
                    dmpnn_morgan_xgb.append([AUC, Sn, Sp, acc])
                xgb_type = 'dmpnn+xgb'
                dmpnn_xgb = [i for i in np.mean(dmpnn_xgb, axis=0)]
                scores.append([xgb_type, max_depth_number, learning_rate_number, min_child_weight_number,dmpnn_xgb[0],dmpnn_xgb[1],dmpnn_xgb[2],dmpnn_xgb[3]])
                xgb_type = 'morgan+xgb'
                morgan_xgb = [i for i in np.mean(morgan_xgb, axis=0)]
                scores.append([xgb_type, max_depth_number, learning_rate_number, min_child_weight_number,morgan_xgb[0],morgan_xgb[1],morgan_xgb[2],morgan_xgb[3]])
                xgb_type = 'dmpnn+morgan+xgb'
                dmpnn_morgan_xgb = [i for i in np.mean(dmpnn_morgan_xgb, axis=0)]
                scores.append([xgb_type, max_depth_number, learning_rate_number, min_child_weight_number,dmpnn_morgan_xgb[0],dmpnn_morgan_xgb[1],dmpnn_morgan_xgb[2],dmpnn_morgan_xgb[3]])
    scores_df = pd.DataFrame(scores)
    return scores_df


def xgb_regre_cv(max_depth_numbers,learning_rate_numbers,min_child_weight_numbers,
               train_feature, train_target,val_feature, val_target,test_feature,test_target,
               train_morgan_feature,val_morgan_feature,test_morgan_feature,test_preds,scaler):
    scores = []
    train_gcn_mor_feature = pd.concat([train_feature, train_morgan_feature], axis=1)
    val_gcn_mor_feature = pd.concat([val_feature, val_morgan_feature], axis=1)
    test_gcn_mor_feature = pd.concat([test_feature, test_morgan_feature], axis=1)
    train_gcn_mor_feature.columns = val_gcn_mor_feature.columns = test_gcn_mor_feature.columns = range(
        train_gcn_mor_feature.shape[1])
    MSE = mean_squared_error(test_target, [i[0] for i in test_preds])
    RMSE = MSE ** 0.5
    MAE = median_absolute_error(test_target, [i[0] for i in test_preds])
    xgb_type = 'dmpnn'
    scores.append([xgb_type, 'none', 'none', 'none', RMSE, MAE])
    for max_depth_number in max_depth_numbers:
        for learning_rate_number in learning_rate_numbers:
            for min_child_weight_number in min_child_weight_numbers:
                xgb_gbc = xgb.XGBRegressor(learn_rate=learning_rate_number, max_depth=max_depth_number, min_child_weight=min_child_weight_number, gamma=1, subsample=0.8,
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
                xgb_type = 'dmpnn+xgb'
                scores.append([xgb_type,max_depth_number,learning_rate_number,min_child_weight_number,RMSE, MAE])
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
                xgb_type = 'morgan+xgb'
                scores.append([xgb_type,max_depth_number,learning_rate_number,min_child_weight_number,RMSE, MAE])
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
                xgb_type = 'dmpnn+morgan+xgb'
                scores.append([xgb_type,max_depth_number,learning_rate_number,min_child_weight_number,RMSE, MAE])
    scores_df = pd.DataFrame(scores)
    return scores_df


def xgb_regre_more(max_depth_numbers,learning_rate_numbers,min_child_weight_numbers,
               train_feature, train_target,val_feature, val_target,test_feature,test_target,
               train_morgan_feature,val_morgan_feature,test_morgan_feature,test_preds,scaler):
    scores = []
    dmpnn_xgb = []
    morgan_xgb = []
    dmpnn_morgan_xgb = []
    train_gcn_mor_feature = pd.concat([train_feature, train_morgan_feature], axis=1)
    val_gcn_mor_feature = pd.concat([val_feature, val_morgan_feature], axis=1)
    test_gcn_mor_feature = pd.concat([test_feature, test_morgan_feature], axis=1)
    train_gcn_mor_feature.columns = val_gcn_mor_feature.columns = test_gcn_mor_feature.columns = range(
        train_gcn_mor_feature.shape[1])
    MSE = mean_squared_error(test_target, test_preds)
    RMSE = MSE ** 0.5
    MAE = median_absolute_error(test_target, test_preds)
    xgb_type = 'dmpnn'
    scores.append([xgb_type, 'none', 'none', 'none', RMSE, MAE])
    for max_depth_number in max_depth_numbers:
        for learning_rate_number in learning_rate_numbers:
            for min_child_weight_number in min_child_weight_numbers:
                for i in range(test_target.shape[1]):
                    xgb_gbc = xgb.XGBRegressor(learn_rate=learning_rate_number, max_depth=max_depth_number, min_child_weight=min_child_weight_number, gamma=1, subsample=0.8,
                                               colsample_bytree=0.8,
                                               reg_alpha=0.8, objective='reg:linear', n_estimators=2000, tree_method='gpu_hist',
                                               n_gpus=-1)
                    xgb_gbc.fit(train_feature, train_target[i], eval_set=[(val_feature, val_target[i])], eval_metric='rmse',
                                early_stopping_rounds=200)
                    y_pred = xgb_gbc.predict(test_feature)
                    y_pred = scaler.inverse_transform(y_pred)
                    y_test = test_target[i].astype('float')
                    MSE = mean_squared_error(y_test, y_pred)
                    RMSE = MSE ** 0.5
                    MAE = median_absolute_error(y_test, y_pred)
                    dmpnn_xgb.append([RMSE, MAE])
                    xgb_gbc = xgb.XGBRegressor(learn_rate=0.1, max_depth=4, min_child_weight=10, gamma=1, subsample=0.8,
                                               colsample_bytree=0.8,
                                               reg_alpha=0.8, objective='reg:linear', n_estimators=2000, tree_method='gpu_hist',
                                               n_gpus=-1)
                    xgb_gbc.fit(train_morgan_feature, train_target[i], eval_set=[(val_morgan_feature, val_target[i])], eval_metric='rmse',
                                early_stopping_rounds=200)
                    y_pred = xgb_gbc.predict(test_morgan_feature)
                    y_pred = scaler.inverse_transform(y_pred)
                    MSE = mean_squared_error(y_test, y_pred)
                    RMSE = MSE ** 0.5
                    MAE = median_absolute_error(y_test, y_pred)
                    morgan_xgb.append([RMSE, MAE])
                    xgb_gbc = xgb.XGBRegressor(learn_rate=0.1, max_depth=4, min_child_weight=10, gamma=1, subsample=0.8,
                                               colsample_bytree=0.8,
                                               reg_alpha=0.8, objective='reg:linear', n_estimators=2000, tree_method='gpu_hist',
                                               n_gpus=-1)
                    xgb_gbc.fit(train_gcn_mor_feature, train_target[i], eval_set=[(val_gcn_mor_feature, val_target[i])], eval_metric='rmse',
                                early_stopping_rounds=200)
                    y_pred = xgb_gbc.predict(test_gcn_mor_feature)
                    y_pred = scaler.inverse_transform(y_pred)
                    MSE = mean_squared_error(y_test, y_pred)
                    RMSE = MSE ** 0.5
                    MAE = median_absolute_error(y_test, y_pred)
                    dmpnn_morgan_xgb.append([RMSE, MAE])
                xgb_type = 'dmpnn+xgb'
                dmpnn_xgb = [i for i in np.mean(dmpnn_xgb, axis=0)]
                scores.append([xgb_type, max_depth_number, learning_rate_number, min_child_weight_number, dmpnn_xgb[0], dmpnn_xgb[1]])
                xgb_type = 'morgan+xgb'
                morgan_xgb = [i for i in np.mean(morgan_xgb, axis=0)]
                scores.append([xgb_type, max_depth_number, learning_rate_number, min_child_weight_number, morgan_xgb[0],morgan_xgb[1]])
                xgb_type = 'dmpnn+morgan+xgb'
                dmpnn_morgan_xgb = [i for i in np.mean(dmpnn_morgan_xgb, axis=0)]
                scores.append([xgb_type, max_depth_number, learning_rate_number, min_child_weight_number, dmpnn_morgan_xgb[0],dmpnn_morgan_xgb[1]])
    scores_df = pd.DataFrame(scores)
    return scores_df
