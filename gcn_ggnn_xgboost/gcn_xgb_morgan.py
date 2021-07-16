
import numpy as np
import torch
import argparse
import os
import pandas as pd

from sklearn.metrics import roc_curve, auc, mean_squared_error
from tensorboardX import SummaryWriter
from torch import optim, nn

from data_deal import decrease_learning_rate
from gcn_model.gcn_model import GCN
from gcn_model.gcn_xgboost_scores import get_feature
from gcn_model.gcn_training import training,evaluate,training_classing,evaluate_classion,evaluate_test_scros
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from load_data_batch import loadInputs_train, loadInputs_val, loadInputs_test, ToxicDataset, loadInputs_feature_smiles

writer = SummaryWriter('loss')

def mechine_regression(X_train,y_train,X_val, y_val,X_test,y_test):
    scores = []
    if y_test.shape[-1]==1:
        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_test = y_test.astype('float')
        MSE = mean_squared_error(y_test,y_pred)
        RMSE = MSE ** 0.5
        type = 'GCN+rf'
        scores.append([type,RMSE])
        clf = svm.SVR(C=0.8, cache_size=200, kernel='rbf', degree=3, epsilon=0.2)
        clf.fit(X_train, y_train)
        y_pre = clf.predict(X_test)
        MSE = mean_squared_error(y_test, y_pre)
        RMSE = MSE ** 0.5
        type = 'GCN+svm'
        scores.append([type, RMSE])
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pre = knn.predict(X_test)
        MSE = mean_squared_error(y_test, y_pre)
        RMSE = MSE ** 0.5
        type = 'GCN+knn'
        scores.append([type, RMSE])
        scores_df = pd.DataFrame(scores)
        return scores_df
    else:
        if len(y_train.shape)==3:
            y_train = [x[0] for x in y_train]
            y_val = [x[0] for x in y_val]
            y_test = [x[0] for x in y_test]
            y_train = pd.DataFrame(y_train)
            y_val = pd.DataFrame(y_val)
            y_test = pd.DataFrame(y_test)
        rf_rmse = []
        svm_rmse = []
        knn_rmse = []
        for i in range(y_test.shape[1]):
            if float(max(y_val[i])) == 0 or float(max(y_train[i])) == 0 or float(max(y_test[i])) == 0:
                continue
            rf = RandomForestRegressor()
            rf.fit(X_train, y_train[i])
            y_pred = rf.predict(X_test)
            y_test = y_test[i].astype('float')
            MSE = mean_squared_error(y_test[i], y_pred)
            RMSE = MSE ** 0.5
            rf_rmse.append(RMSE)
        type = 'GCN+rf'
        scores.append([type, np.mean(rf_rmse)])
        for i in range(y_test.shape[1]):
            clf = svm.SVR(C=0.8, cache_size=200, kernel='rbf', degree=3, epsilon=0.2)
            clf.fit(X_train, y_train[i])
            y_pre = clf.predict(X_test)
            MSE = mean_squared_error(y_test[i], y_pre)
            RMSE = MSE ** 0.5
            type = 'GCN+svm'
            scores.append([type, RMSE])
            knn = KNeighborsRegressor(n_neighbors=5)
            knn.fit(X_train, y_train[i])
            y_pre = knn.predict(X_test)
            MSE = mean_squared_error(y_test[i], y_pre)
            RMSE = MSE ** 0.5
            svm_rmse.append([type, RMSE])
        type = 'GCN+svm'
        scores.append([type, np.mean(svm_rmse)])
        for i in range(y_test.shape[1]):
            knn = KNeighborsRegressor(n_neighbors=5)
            knn.fit(X_train, y_train[i])
            y_pre = knn.predict(X_test)
            MSE = mean_squared_error(y_test[i], y_pre)
            RMSE = MSE ** 0.5
            knn_rmse.append(RMSE)
        type = 'GCN+knn'
        scores.append([type, np.mean(knn_rmse)])
        scores_df = pd.DataFrame(scores)
        return scores_df



def mechine_classion(X_train,y_train,X_val, y_val,X_test,y_test):
    if y_test.shape[-1]==1:
        scores = []
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)  # 使用训练集对测试集进行训练
        y_pre = rf.predict(X_test)
        fpr, tpr, threshold = roc_curve(y_test, y_pre)
        AUC = auc(fpr, tpr)
        type = 'GCN+rf'
        scores.append([type, AUC])
        clf = svm.SVC(C=1, kernel='rbf', degree=3, gamma='auto',
                      coef0=0.0, shrinking=True, probability=False,
                      tol=1e-3, cache_size=200, class_weight=None,
                      verbose=False, max_iter=-1, decision_function_shape='ovr',
                      random_state=None)
        clf.fit(X_train, y_train)
        y_pre = clf.predict(X_test)
        fpr, tpr, threshold = roc_curve(y_test, y_pre)
        AUC = auc(fpr, tpr)
        type = 'GCN+svm'
        scores.append([type, AUC])
        clf = KNeighborsClassifier(n_neighbors=5, weights='uniform')
        clf.fit(X_train, y_train)
        y_pre = clf.predict(X_test)
        fpr, tpr, threshold = roc_curve(y_test, y_pre)
        AUC = auc(fpr, tpr)
        type = 'GCN+rnn'
        scores.append([type, AUC])
        scores_df = pd.DataFrame(scores)
        return scores_df
    else:
        rf_auc = []
        svm_auc = []
        knn_auc = []
        scores = []
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
            rf = RandomForestClassifier()
            rf.fit(X_train, y_train[i])  # 使用训练集对测试集进行训练
            y_pre = rf.predict(X_test)
            fpr, tpr, threshold = roc_curve([float(j) for j in y_test[i]], [float(k) for k in y_pre])
            AUC = auc(fpr, tpr)
            rf_auc.append(AUC)
        type = 'GCN+rf'
        scores.append([type, np.mean(rf_auc)])
        for i in range(y_test.shape[1]):
            clf = svm.SVC(C=1, kernel='rbf', degree=3, gamma='auto',
                          coef0=0.0, shrinking=True, probability=False,
                          tol=1e-3, cache_size=200, class_weight=None,
                          verbose=False, max_iter=-1, decision_function_shape='ovr',
                          random_state=None)
            clf.fit(X_train, y_train[i])
            y_pre = clf.predict(X_test)
            fpr, tpr, threshold = roc_curve([float(j) for j in y_test[i]], [float(k) for k in y_pre])
            AUC = auc(fpr, tpr)
            if AUC>0:
                svm_auc.append(AUC)
        type = 'GCN+svm'
        scores.append([type, np.mean(svm_auc)])
        for i in range(y_test.shape[1]):
            clf = KNeighborsClassifier(n_neighbors=5, weights='uniform')
            clf.fit(X_train, y_train[i])
            y_pre = clf.predict(X_test)
            fpr, tpr, threshold = roc_curve([float(j) for j in y_test[i]], [float(k) for k in y_pre])
            AUC = auc(fpr, tpr)
            if AUC>0:
                knn_auc.append(AUC)
        type = 'GCN+knn'
        scores.append([type, np.mean(knn_auc)])
        scores_df = pd.DataFrame(scores)
        return scores_df

def get_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10,#800
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--folder', type=str, default="benchmark_molnet/molnet_random_bace_c")
    parser.add_argument('--hidden', type=int, default=512,#64
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batch_size', type=float, default=128)
    parser.add_argument('--natom', type=float, default=58)
    parser.add_argument('--protein',type=str,default='bbbp')
    parser.add_argument('--nclass', type=float, default=1)
    parser.add_argument('--type', type=str, default="classification")
    args = parser.parse_args()
    return args


def mechine_scores(gcn_scores,xgb_scores,names,args,device):
    for i in range(3):
        args.dataset = args.folder + '/seed' + str(i + 1)
        feature_train,a_train,y_train,smiles_train = loadInputs_feature_smiles(args,'train')
        feature_val,a_val,y_val,smiles_val = loadInputs_feature_smiles(args,'val')
        feature_test,a_test,y_test,smiles_test = loadInputs_feature_smiles(args,'test')
        args.nclass = y_test.shape[-1]
        names['model' + str(i) ] = GCN(args.natom, args.hidden, args.nclass, args.dropout).to(device)
        optimizer = optim.Adam(names['model' + str(i) ].parameters(), lr=args.lr)
        train_dataset = ToxicDataset(feature_train, a_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataset = ToxicDataset(feature_val, a_val, y_val)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_dataset = ToxicDataset(feature_test, a_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        if args.type == 'regression':
            args.metric = 'RMSE'
            criterion = nn.MSELoss()
            for epoch in range(args.epochs):
                train_loss, train_R2, val_total_loss = training(names['model' + str(i)], train_loader, optimizer,
                                                                criterion, device)
                val_loss, val_RMSE, val_R2, _,_ = evaluate(names['model' + str(i)], val_loader, criterion, device)
                if epoch % 4 == 0 and epoch != 0:
                    decrease_learning_rate(optimizer, decrease_by=0.001)
            test_loss, test_RMSE, test_R2, _,MAE = evaluate(names['model' + str(i)], test_loader, criterion, device)
            train_dataset = ToxicDataset(feature_train, a_train, y_train)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
            xgb_train_feature = get_feature(names['model' + str(i)], train_loader, device)
            xgb_val_feature = get_feature(names['model' + str(i)], val_loader, device)
            xgb_test_feature = get_feature(names['model' + str(i)], test_loader, device)
            xgb_train_feature = xgb_train_feature.cpu().numpy()
            xgb_val_feature = xgb_val_feature.cpu().numpy()
            xgb_test_feature = xgb_test_feature.cpu().numpy()
            rmse_df = mechine_regression(xgb_train_feature, y_train, xgb_val_feature, y_val, xgb_test_feature, y_test)
            return rmse_df
        else:
            args.metric = 'AUC'
            criterion = nn.BCEWithLogitsLoss()
            for epoch in range(args.epochs):
                train_loss, val_total_loss = training_classing(names['model' + str(i)], train_loader, optimizer,
                                                               criterion, device)
                val_loss, val_AUC, val_precision, val_recall = evaluate_classion(names['model' + str(i)], val_loader,
                                                                                 criterion, device)

                if epoch % 4 == 0 and epoch != 0:
                    decrease_learning_rate(optimizer, decrease_by=0.001)
            test_loader_xgb = (torch.from_numpy(np.float32(feature_test)), torch.from_numpy(np.float32(a_test)),
                               torch.from_numpy(np.float32(y_test)))
            test_loss, test_AUC, test_precision, test_recall = evaluate_test_scros(names['model' + str(i)],
                                                                                   test_loader_xgb, criterion, device)
            train_dataset = ToxicDataset(feature_train, a_train, y_train)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
            xgb_train_feature = get_feature(names['model' + str(i)], train_loader, device)
            xgb_val_feature = get_feature(names['model' + str(i)], val_loader, device)
            xgb_test_feature = get_feature(names['model' + str(i)], test_loader, device)
            xgb_train_feature = xgb_train_feature.cpu().numpy()
            xgb_val_feature = xgb_val_feature.cpu().numpy()
            xgb_test_feature = xgb_test_feature.cpu().numpy()
            auc_df = mechine_classion(xgb_train_feature, y_train, xgb_val_feature, y_val, xgb_test_feature, y_test)
            return auc_df


# if __name__=="__main__":
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#     args = get_argv()
#     args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     gcn_scores = []
#     xgb_gcn = []
#     gcn_morgan = []
#     morgan_scores = []
#     gcn_scores,xgb_gcn,gcn_morgan,morgan_scores = get_scores(gcn_scores,xgb_gcn,gcn_morgan,morgan_scores,args)
#     writer.close()
#     gcn_scores,xgb_gcn,gcn_morgan,morgan_scores = np.array(gcn_scores),np.array(xgb_gcn),np.array(gcn_morgan),np.array(morgan_scores)
#
#     gcn_scores = np.nanmean(gcn_scores, axis=1)  # average score for each model across tasks
#     mean_score, std_score = np.nanmean(gcn_scores), np.nanstd(gcn_scores)
#     print(f'Overall gcn test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')
#
#     xgb_gcn = np.nanmean(xgb_gcn, axis=1)  # average score for each model across tasks
#     xgb_mean_score, xgb_std_score = np.nanmean(xgb_gcn), np.nanstd(xgb_gcn)
#     print(f'Overall xgb_gcn_auc {args.metric} = {xgb_mean_score:.6f} +/- {xgb_std_score:.6f}')
#
#     morgan_scores = np.nanmean(morgan_scores, axis=1)  # average score for each model across tasks
#     morgan_mean_score, morgan_std_score = np.nanmean(morgan_scores), np.nanstd(morgan_scores)
#     print(f'Overall morgan_test {args.metric} = {morgan_mean_score:.6f} +/- {morgan_std_score:.6f}')
#
#     gcn_morgan = np.nanmean(gcn_morgan, axis=1)  # average score for each model across tasks
#     gcn_morgan_mean_score, gcn_morgan_std_score = np.nanmean(gcn_morgan), np.nanstd(gcn_morgan)
#     print(f'Overall gcn_morgan_auc {args.metric} = {gcn_morgan_mean_score:.6f} +/- {gcn_morgan_std_score:.6f}')

if __name__=="__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_argv()

    gcn_scores = []
    xgb_scores = []
    names = locals()
    df = mechine_scores(gcn_scores,xgb_scores,names,args,device)
    df.to_csv(args.protein + '_gcn_mechine.csv',index=None)