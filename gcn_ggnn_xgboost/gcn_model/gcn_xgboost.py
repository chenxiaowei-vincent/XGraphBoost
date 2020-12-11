import numpy as np
from sklearn.metrics import precision_recall_curve,mean_squared_error,mean_absolute_error
from sklearn.metrics import roc_curve, auc
import xgboost as xgb
import pandas as pd
import torch
from xgboost.sklearn import XGBRegressor



def get_feature(model,data, device):
    model.eval()
    with torch.no_grad():
        feature,A,y = data
        if len(y.shape) == 3:
            y = y.squeeze(1)
        feature, A, y = feature.to(device), A.to(device), y.to(device)
        output,feature = model(feature,A)
    return feature



def xgb_regression(X_train,y_train,X_val, y_val,X_test,y_test,args):
    if y_test.shape[-1] == 1:
        model = XGBRegressor(
            learn_rate=0.1,
            max_depth=4,  # 4
            min_child_weight=10,
            gamma=1,  # 1
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.8,
            objective='reg:linear',
            n_estimators=2000,
            tree_method='gpu_hist',
            n_gpus=-1
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse',
                  early_stopping_rounds=300)
        y_pred = model.predict(X_test)
        y_test = y_test.astype('float')
        MSE = mean_squared_error(y_test, y_pred)
        RMSE = MSE ** 0.5
        return RMSE
    else:
        RMSEs = []
        if len(y_train.shape) == 3:
            y_train = [x[0] for x in y_train]
            y_val = [x[0] for x in y_val]
            y_test = [x[0] for x in y_test]
            y_train = pd.DataFrame(y_train)
            y_val = pd.DataFrame(y_val)
            y_test = pd.DataFrame(y_test)
        for i in range(y_test.shape[1]):
            if float(max(y_val[i])) == 0 or float(max(y_train[i])) == 0 or float(max(y_test[i])) == 0:
                continue
            model = XGBRegressor(
                learn_rate=0.1,
                max_depth=4,  # 4
                min_child_weight=10,
                gamma=1,  # 1
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.8,
                objective='reg:linear',
                n_estimators=2000,
                tree_method='gpu_hist',
                n_gpus=-1
            )
            model.fit(X_train, [float(k) for k in y_train[i]], eval_set=[(X_val, [float(k) for k in y_val[i]])], eval_metric='rmse',
                      early_stopping_rounds=300)
            y_pred = model.predict(X_test)
            y_test = y_test.astype('float')
            MSE = mean_squared_error(y_test[i], y_pred)
            RMSE = MSE ** 0.5
            RMSEs.append(RMSE)
        return np.mean(RMSEs)


def xgboost_classion(X_train,y_train,X_val, y_val,X_test,y_test,args):
    if y_test.shape[-1]==1:
        xgb_gbc = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
               colsample_bytree=1, gamma=1, learning_rate=0.1, max_delta_step=0,
               max_depth=4, min_child_weight=8, missing=None, n_estimators=2000,
               n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
               silent=True, subsample=0.8,tree_method='gpu_hist',n_gpus=-1)
        xgb_gbc.fit(X_train,y_train,eval_set = [(X_val,y_val)],eval_metric = 'auc',early_stopping_rounds=300)
        pre_pro = xgb_gbc.predict_proba(X_test)[:,1]
        fpr,tpr,threshold = roc_curve([float(i) for i in y_test],pre_pro)
        AUC = auc(fpr,tpr)
        return AUC
    else:
        aucs = []
        if len(y_train.shape)==3:
            y_train = [x[0] for x in y_train]
            y_val = [x[0] for x in y_val]
            y_test = [x[0] for x in y_test]
            y_train = pd.DataFrame(y_train)
            y_val = pd.DataFrame(y_val)
            y_test = pd.DataFrame(y_test)
        for i in range(y_test.shape[1]):
            if float(max(y_val[i])) == 0 or float(max(y_train[i])) == 0 or float(max(y_test[i])) == 0:
                continue
            xgb_gbc = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                        colsample_bytree=1, gamma=1, learning_rate=0.1, max_delta_step=0,
                                        max_depth=4, min_child_weight=8, missing=None, n_estimators=2000,
                                        n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
                                        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                                        silent=True, subsample=0.8, tree_method='gpu_hist', n_gpus=-1)
            xgb_gbc.fit(X_train, y_train[i], eval_set=[(X_val, y_val[i])], eval_metric='auc', early_stopping_rounds=300)
            pre_pro = xgb_gbc.predict_proba(X_test)[:, 1]
            fpr, tpr, threshold = roc_curve([float(j) for j in y_test[i]], pre_pro)
            AUC = auc(fpr, tpr)
            aucs.append(AUC)
        return np.mean(aucs)