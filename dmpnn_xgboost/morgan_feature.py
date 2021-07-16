from math import isnan
import pandas as pd
from rdkit.Chem import AllChem
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error,median_absolute_error
from sklearn.metrics import roc_curve, auc,precision_recall_curve,confusion_matrix,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor

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

def svm_knn_rf_class(train_feature, train_target,val_feature, val_target,test_feature,test_target,
                     train_morgan_feature,val_morgan_feature,test_morgan_feature,test_preds):
    scores = []
    rf = RandomForestClassifier()
    rf.fit(train_feature, train_target)  # 使用训练集对测试集进行训练
    y_pre = rf.predict(test_feature)
    fpr, tpr, threshold = roc_curve(test_target, y_pre)
    AUC = auc(fpr, tpr)
    type = 'dmpnn+rf'
    scores.append([type, AUC])
    clf = svm.SVC(C=1, kernel='rbf', degree=3, gamma='auto',
                  coef0=0.0, shrinking=True, probability=False,
                  tol=1e-3, cache_size=200, class_weight=None,
                  verbose=False, max_iter=-1, decision_function_shape='ovr',
                  random_state=None)
    clf.fit(train_feature, train_target)
    y_pre = clf.predict(test_feature)
    fpr, tpr, threshold = roc_curve(test_target, y_pre)
    AUC = auc(fpr, tpr)
    type = 'dmpnn+svm'
    scores.append([type, AUC])
    clf = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    clf.fit(train_feature, train_target)
    y_pre = clf.predict(test_feature)
    fpr, tpr, threshold = roc_curve(test_target, y_pre)
    AUC = auc(fpr, tpr)
    type = 'dmpnn+rnn'
    scores.append([type, AUC])
    scores_df = pd.DataFrame(scores)
    return scores_df

def svm_knn_rf_class_more(train_feature, train_target,val_feature, val_target,test_feature,test_target,
                     train_morgan_feature,val_morgan_feature,test_morgan_feature,test_preds):
    scores = []
    rf_auc = []
    svm_auc = []
    knn_auc = []
    for i in range(test_target.shape[1]):
        rf = RandomForestClassifier()
        rf.fit(train_feature, train_target[i])  # 使用训练集对测试集进行训练
        y_pre = rf.predict(test_feature)
        fpr, tpr, threshold = roc_curve(test_target[i], y_pre)
        AUC = auc(fpr, tpr)
        if AUC>0:
            rf_auc.append(AUC)
    type = 'dmpnn+rf'
    scores.append([type, np.mean(rf_auc)])
    for i in range(test_target.shape[1]):
        clf = svm.SVC(C=1, kernel='rbf', degree=3, gamma='auto',
                      coef0=0.0, shrinking=True, probability=False,
                      tol=1e-3, cache_size=200, class_weight=None,
                      verbose=False, max_iter=-1, decision_function_shape='ovr',
                      random_state=None)
        clf.fit(train_feature, train_target[i])
        y_pre = clf.predict(test_feature)
        fpr, tpr, threshold = roc_curve(test_target[i], y_pre)
        AUC = auc(fpr, tpr)
        if AUC>0:
            svm_auc.append(AUC)
    type = 'dmpnn+svm'
    scores.append([type, np.mean(svm_auc)])
    for i in range(test_target.shape[1]):
        clf = KNeighborsClassifier(n_neighbors=5, weights='uniform')
        clf.fit(train_feature, train_target[i])
        y_pre = clf.predict(test_feature)
        fpr, tpr, threshold = roc_curve(test_target[i], y_pre)
        AUC = auc(fpr, tpr)
        if AUC>0:
            knn_auc.append(AUC)
    type = 'dmpnn+rnn'
    scores.append([type, np.mean(knn_auc)])
    scores_df = pd.DataFrame(scores)
    return scores_df


def svm_knn_rf_regre(train_feature, train_target,val_feature, val_target,test_feature,test_target,
                     train_morgan_feature,val_morgan_feature,test_morgan_feature,test_preds):
    scores = []
    rf = RandomForestRegressor()
    rf.fit(train_feature, train_target)
    y_pre = rf.predict(test_feature)
    MSE = mean_squared_error(test_target, y_pre)
    RMSE = MSE ** 0.5
    type = 'dmpnn+rf'
    scores.append([type,RMSE])
    clf = svm.SVR(C=0.8, cache_size=200, kernel='rbf', degree=3, epsilon=0.2)
    clf.fit(train_feature, train_target)
    y_pre = clf.predict(test_feature)
    MSE = mean_squared_error(test_target, y_pre)
    print("SVM-val-MSE:", MSE)
    RMSE = MSE ** 0.5
    type = 'dmpnn+svm'
    scores.append([type, RMSE])
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(train_feature, train_target)
    y_pre = knn.predict(test_feature)
    MSE = mean_squared_error(test_target, y_pre)
    RMSE = MSE ** 0.5
    type = 'dmpnn+knn'
    scores.append([type, RMSE])
    scores_df = pd.DataFrame(scores)
    return scores_df

def svm_knn_rf_regre_more(train_feature, train_target,val_feature, val_target,test_feature,test_target,
                     train_morgan_feature,val_morgan_feature,test_morgan_feature,test_preds):
    scores = []
    rf_rmse = []
    svm_rmse = []
    knn_rmse = []
    for i in range(test_target.shape[1]):
        rf = RandomForestRegressor()
        rf.fit(train_feature, train_target)
        y_pre = rf.predict(test_feature)
        MSE = mean_squared_error(test_target, y_pre)
        RMSE = MSE ** 0.5
        rf_rmse.append(RMSE)
    rf_rmse = [i for i in np.mean(rf_rmse, axis=0)]
    type = 'dmpnn+rf'
    scores.append([type,rf_rmse])
    for i in range(test_target.shape[1]):
        clf = svm.SVR(C=0.8, cache_size=200, kernel='rbf', degree=3, epsilon=0.2)
        clf.fit(train_feature, train_target)
        y_pre = clf.predict(test_feature)
        MSE = mean_squared_error(test_target, y_pre)
        RMSE = MSE ** 0.5
        svm_rmse.append(RMSE)
    svm_rmse = [i for i in np.mean(svm_rmse, axis=0)]
    type = 'dmpnn+svm'
    scores.append([type, svm_rmse])
    for i in range(test_target.shape[1]):
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(train_feature, train_target)
        y_pre = knn.predict(test_feature)
        MSE = mean_squared_error(test_target, y_pre)
        RMSE = MSE ** 0.5
        knn_rmse.append(RMSE)
    knn_rmse = [i for i in np.mean(knn_rmse, axis=0)]
    type = 'dmpnn+knn'
    scores.append([type, knn_rmse])
    scores_df = pd.DataFrame(scores)
    return scores_df

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
    xgb_type = 'dmpnn'
    scores.append([xgb_type, 'none', 'none', 'none', RMSE])
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
                y_test = test_target.astype('float')
                MSE = mean_squared_error(y_test, y_pred)
                RMSE = MSE ** 0.5
                xgb_type = 'dmpnn+xgb'
                scores.append([xgb_type,max_depth_number,learning_rate_number,min_child_weight_number,RMSE])
                xgb_gbc = xgb.XGBRegressor(learn_rate=0.1, max_depth=4, min_child_weight=10, gamma=1, subsample=0.8,
                                           colsample_bytree=0.8,
                                           reg_alpha=0.8, objective='reg:linear', n_estimators=2000, tree_method='gpu_hist',
                                           n_gpus=-1)
                xgb_gbc.fit(train_morgan_feature, train_target, eval_set=[(val_morgan_feature, val_target)], eval_metric='rmse',
                            early_stopping_rounds=200)
                y_pred = xgb_gbc.predict(test_morgan_feature)
                MSE = mean_squared_error(y_test, y_pred)
                RMSE = MSE ** 0.5
                xgb_type = 'morgan+xgb'
                scores.append([xgb_type,max_depth_number,learning_rate_number,min_child_weight_number,RMSE])
                xgb_gbc = xgb.XGBRegressor(learn_rate=0.1, max_depth=4, min_child_weight=10, gamma=1, subsample=0.8,
                                           colsample_bytree=0.8,
                                           reg_alpha=0.8, objective='reg:linear', n_estimators=2000, tree_method='gpu_hist',
                                           n_gpus=-1)
                xgb_gbc.fit(train_gcn_mor_feature, train_target, eval_set=[(val_gcn_mor_feature, val_target)], eval_metric='rmse',
                            early_stopping_rounds=200)
                y_pred = xgb_gbc.predict(test_gcn_mor_feature)
                MSE = mean_squared_error(y_test, y_pred)
                RMSE = MSE ** 0.5
                xgb_type = 'dmpnn+morgan+xgb'
                scores.append([xgb_type,max_depth_number,learning_rate_number,min_child_weight_number,RMSE])
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
    xgb_type = 'dmpnn'
    scores.append([xgb_type, 'none', 'none', 'none', RMSE])
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
                    y_test = test_target[i].astype('float')
                    MSE = mean_squared_error(y_test, y_pred)
                    RMSE = MSE ** 0.5
                    dmpnn_xgb.append([RMSE])
                    xgb_gbc = xgb.XGBRegressor(learn_rate=0.1, max_depth=4, min_child_weight=10, gamma=1, subsample=0.8,
                                               colsample_bytree=0.8,
                                               reg_alpha=0.8, objective='reg:linear', n_estimators=2000, tree_method='gpu_hist',
                                               n_gpus=-1)
                    xgb_gbc.fit(train_morgan_feature, train_target[i], eval_set=[(val_morgan_feature, val_target[i])], eval_metric='rmse',
                                early_stopping_rounds=200)
                    y_pred = xgb_gbc.predict(test_morgan_feature)
                    MSE = mean_squared_error(y_test, y_pred)
                    RMSE = MSE ** 0.5
                    morgan_xgb.append([RMSE])
                    xgb_gbc = xgb.XGBRegressor(learn_rate=0.1, max_depth=4, min_child_weight=10, gamma=1, subsample=0.8,
                                               colsample_bytree=0.8,
                                               reg_alpha=0.8, objective='reg:linear', n_estimators=2000, tree_method='gpu_hist',
                                               n_gpus=-1)
                    xgb_gbc.fit(train_gcn_mor_feature, train_target[i], eval_set=[(val_gcn_mor_feature, val_target[i])], eval_metric='rmse',
                                early_stopping_rounds=200)
                    y_pred = xgb_gbc.predict(test_gcn_mor_feature)
                    MSE = mean_squared_error(y_test, y_pred)
                    RMSE = MSE ** 0.5
                    dmpnn_morgan_xgb.append([RMSE])
                xgb_type = 'dmpnn+xgb'
                dmpnn_xgb = [i for i in np.mean(dmpnn_xgb, axis=0)]
                scores.append([xgb_type, max_depth_number, learning_rate_number, min_child_weight_number, dmpnn_xgb[0]])
                xgb_type = 'morgan+xgb'
                morgan_xgb = [i for i in np.mean(morgan_xgb, axis=0)]
                scores.append([xgb_type, max_depth_number, learning_rate_number, min_child_weight_number, morgan_xgb[0]])
                xgb_type = 'dmpnn+morgan+xgb'
                dmpnn_morgan_xgb = [i for i in np.mean(dmpnn_morgan_xgb, axis=0)]
                scores.append([xgb_type, max_depth_number, learning_rate_number, min_child_weight_number, dmpnn_morgan_xgb[0]])
    scores_df = pd.DataFrame(scores)
    return scores_df
