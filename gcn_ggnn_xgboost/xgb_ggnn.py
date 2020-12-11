
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import Dataset
from load_data_batch import loadInputs_train,loadInputs_val,loadInputs_test,ToxicDataset,load_data
from sklearn.metrics import precision_recall_curve,mean_squared_error,r2_score,mean_absolute_error
from sklearn.metrics import roc_curve, auc, precision_score, recall_score
import xgboost as xgb
from ggnn_model.ggnn_model import GGNN
import os
from data_deal import decrease_learning_rate
import pandas as pd
from torch.autograd import Variable
from tensorboardX import SummaryWriter


writer = SummaryWriter('loss')




def training(model, data,optimizer, criterion, args):
    model.train()
    total_loss = []
    for k in data:
        feature,A,y = k
        if len(y.shape) == 3:
            y = y.squeeze(1)
        model.zero_grad()
        padding = torch.zeros(len(feature), 50, args.hidden - 58)
        init_input = torch.cat((feature, padding), 2)
        init_input,A,feature,y = init_input.cuda(),A.cuda(),feature.cuda(),y.cuda()
        init_input,A,feature = Variable(init_input),Variable(A),Variable(feature)
        target = Variable(y)
        output,_ = model(init_input, feature, A)
        loss = criterion(output, target)

        total_loss.append(loss.item())
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        R2 = r2_score(y.cpu().detach().numpy(), output.cpu().detach().numpy())
    model.eval()
    val_total_loss = 0
    with torch.no_grad():
        for k in data:
            feature, A, y = k
            if len(y.shape) == 3:
                y = y.squeeze(1)
            padding = torch.zeros(len(feature), 50, args.hidden - 58)
            init_input = torch.cat((feature, padding), 2)
            init_input, A, feature, y = Variable(init_input.cuda()), Variable(A.cuda()), Variable(
                feature.cuda()), Variable(y.cuda())
            output, _ = model(init_input, feature, A)
            valid_loss = criterion(output, y)
            val_total_loss = val_total_loss+valid_loss
    return (sum(total_loss) / len(total_loss)),R2,val_total_loss

def evaluate(model, data, criterion, args):
    model.eval()
    total_loss = []
    with torch.no_grad():
        for k in data:
            feature, A, y = k
            if len(y.shape) == 3:
                y = y.squeeze(1)
            padding = torch.zeros(len(feature), 50, args.hidden - 58)
            init_input = torch.cat((feature, padding), 2)
            init_input,A ,feature,y= Variable(init_input.cuda()),Variable(A.cuda()),Variable(feature.cuda()),Variable(y.cuda())
            output, feature = model(init_input, feature, A)
            total_loss.append((criterion(output,  y)).item())
            MSE = mean_squared_error(y.cpu().numpy(), output.cpu().numpy())
            RMSE = MSE ** 0.5
            R2 = r2_score(y.cpu().numpy(), output.cpu().numpy())
            mae = mean_absolute_error(y.cpu().numpy(), output.cpu().numpy())
    return (sum(total_loss) / len(total_loss)),RMSE,R2,feature,mae

def evaluate_test_scores(model, data, criterion, args):
    model.eval()
    total_loss = []
    with torch.no_grad():
        feature, A, y = data
        if len(y.shape) == 3:
            y = y.squeeze(1)
        padding = torch.zeros(len(feature), 50, args.hidden - 58)
        init_input = torch.cat((feature, padding), 2)
        init_input, A, feature, y = Variable(init_input.cuda()), Variable(A.cuda()), Variable(feature.cuda()), Variable(
            y.cuda())
        output, feature = model(init_input, feature, A)
        total_loss.append((criterion(output, y)).item())
        MSE = mean_squared_error(y.cpu().numpy(), output.cpu().numpy())
        RMSE = MSE ** 0.5
    return RMSE


def get_feature(model,data, args):
    model.eval()
    i = 0
    with torch.no_grad():
        for k in data:
            feature, A, y = k
            padding = torch.zeros(len(feature), 50, args.hidden - 58)
            init_input = torch.cat((feature, padding), 2)
            init_input, A, feature, y = Variable(init_input.cuda()), Variable(A.cuda()), Variable(feature.cuda()), Variable(y.cuda())
            output, feature = model(init_input, feature, A)
            if i ==0:
                features = feature
            else:
                features = torch.cat((features,feature))
            i = i+1
    return features

def training_classing(model, data,optimizer, criterion, args):
    model.train()
    total_loss = []
    for k in data:
        feature,A,y = k
        model.zero_grad()
        if len(y.shape)==3:
            y = y.squeeze(1)
        padding = torch.zeros(len(feature), 50, args.hidden - 58)
        init_input = torch.cat((feature, padding), 2)
        init_input = init_input.cuda()
        init_input, A, feature, y = Variable(init_input.cuda()), Variable(A.cuda()), Variable(
            feature.cuda()), Variable(y.cuda())
        output, _ = model(init_input, feature, A)
        loss = criterion(output, y)
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    model.eval()
    val_total_loss = 0
    with torch.no_grad():
        for k in data:
            feature, A, y = k
            if len(y.shape) == 3:
                y = y.squeeze(1)
            padding = torch.zeros(len(feature), 50, args.hidden - 58)
            init_input = torch.cat((feature, padding), 2)
            init_input, A, feature, y = Variable(init_input.cuda()), Variable(A.cuda()), Variable(
                feature.cuda()), Variable(y.cuda())
            output, _ = model(init_input, feature, A)
            valid_loss = criterion(output, y)
            val_total_loss = val_total_loss+valid_loss
    return (sum(total_loss) / len(total_loss)),val_total_loss

def evaluate_classion(model, data, criterion, args):
    model.eval()
    total_loss = []
    y_predict = []
    y_test = []
    with torch.no_grad():
        for k in data:
            feature, A, y = k
            if len(y.shape) == 3:
                y = y.squeeze(1)
            padding = torch.zeros(len(feature), 50, args.hidden - 58)
            init_input = torch.cat((feature, padding), 2)
            init_input, A, feature, y = Variable(init_input.cuda()), Variable(A.cuda()), Variable(
                feature.cuda()), Variable(y.cuda())
            output, _ = model(init_input, feature, A)
            total_loss.append((criterion(output,  y)).item())
            output = torch.sigmoid(output)
            output = output.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            for i in output:
                y_predict.append(i)
            for j in y:
                y_test.append(j)
    y_test = pd.DataFrame(y_test)
    y_predict = pd.DataFrame(y_predict)
    if y_test.shape[1]==1:
        fpr, tpr, threshold = roc_curve(y_test, y_predict)
        AUC = auc(fpr, tpr)
        output_tran = []
        for x in y_predict[0]:
            if x > 0.5:
                output_tran.append(1)
            else:
                output_tran.append(0)
        precision = precision_score(y_test, output_tran)
        recall = recall_score(y_test, output_tran)
    else:
        AUC_all = []
        precision_all = []
        recall_all = []
        for i in range(y_test.shape[1]):
            if max(y_test[i])==0:
                continue
            fpr, tpr, threshold = roc_curve(y_test[i], y_predict[i])
            AUC = auc(fpr, tpr)
            output_tran = []
            for x in y_predict[i]:
                if x > 0.5:
                    output_tran.append(1)
                else:
                    output_tran.append(0)
            precision = precision_score(y_test[i], output_tran)
            recall = recall_score(y_test[i], output_tran)
            AUC_all.append(AUC)
            precision_all.append(precision)
            recall_all.append(recall)
        AUC = np.mean(AUC_all)
        precision = np.mean(precision_all)
        recall = np.mean(recall_all)
    return (sum(total_loss) / len(total_loss)),AUC,precision,recall


def evaluate_test_scros(model, data, criterion, args):
    model.eval()
    total_loss = []
    y_predict = []
    y_test = []
    with torch.no_grad():
        feature, A, y = data
        if len(y.shape)==3:
            y = y.squeeze(1)
        padding = torch.zeros(len(feature), 50, args.hidden - 58)
        init_input = torch.cat((feature, padding), 2)
        init_input, A, feature, y = Variable(init_input.cuda()), Variable(A.cuda()), Variable(
            feature.cuda()), Variable(y.cuda())
        output, _ = model(init_input, feature, A)
        total_loss.append((criterion(output,  y)).item())
        output = torch.sigmoid(output)
        output = output.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        for i in output:
            y_predict.append(i)
        for j in y:
            y_test.append(j)
    y_test = pd.DataFrame(y_test)
    y_predict = pd.DataFrame(y_predict)
    if y_test.shape[1]==1:
        fpr, tpr, threshold = roc_curve(y_test, y_predict)
        AUC = auc(fpr, tpr)
        output_tran = []
        for x in y_predict[0]:
            if x > 0.5:
                output_tran.append(1)
            else:
                output_tran.append(0)
        precision = precision_score(y_test, output_tran)
        recall = recall_score(y_test, output_tran)
    else:
        AUC_all = []
        precision_all = []
        recall_all = []
        for i in range(y_test.shape[1]):
            if max(y_test[i])==0 or max(y_predict[i])==0:
                continue
            fpr, tpr, threshold = roc_curve(y_test[i], y_predict[i])
            AUC = auc(fpr, tpr)
            output_tran = []
            for x in y_predict[i]:
                if x > 0.5:
                    output_tran.append(1)
                else:
                    output_tran.append(0)
            precision = precision_score(y_test[i], output_tran)
            recall = recall_score(y_test[i], output_tran)
            if args.folder == "benchmark_molnet/molnet_random_pcba_c" or args.folder == "benchmark_molnet/molnet_random_muv_c":
                precision, recall, _ = precision_recall_curve(y_test[i], output_tran)
                AUC = auc(recall, precision)
            AUC_all.append(AUC)
            precision_all.append(precision)
            recall_all.append(recall)
        AUC = np.mean(AUC_all)
        precision = np.mean(precision_all)
        recall = np.mean(recall_all)
    return (sum(total_loss) / len(total_loss)),AUC,precision,recall

def xgb_regression(X_train,y_train,X_val, y_val,X_test,y_test):
    from xgboost.sklearn import XGBRegressor
    if y_test.shape[-1]==1:
        model = XGBRegressor(
            learn_rate=0.1,
            max_depth=4,#4
            min_child_weight=10,
            gamma=1,#1
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.8,
            objective='reg:linear',
            n_estimators=2000,
            tree_method = 'gpu_hist',
            n_gpus = -1
        )
        model.fit(X_train, y_train,eval_set=[(X_val, y_val)], eval_metric='rmse',
                  early_stopping_rounds=300)
        y_pred = model.predict(X_test)
        y_test = y_test.astype('float')
        MSE = mean_squared_error(y_test,y_pred)
        RMSE = MSE ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        if args.folder == "benchmark_molnet/molnet_random_qm7_r":
            return mae
        else:
            return RMSE
    else:
        RMSEs = []
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
            mae = mean_absolute_error(y_test[i], y_pred)
            if args.folder == "benchmark_molnet/molnet_random_qm8_r" or args.folder == "benchmark_molnet/molnet_random_qm9_r":
                RMSEs.append(mae)
            else:
                RMSEs.append(RMSE)
        return np.mean(RMSEs)



def xgboost_classion(X_train,y_train,X_val, y_val,X_test,y_test):
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
        if args.folder == "benchmark_molnet/molnet_random_pcba_c" or args.folder == "benchmark_molnet/molnet_random_muv_c":
            precision, recall, _ = precision_recall_curve(y_test, pre_pro)
            AUC = auc(recall, precision)
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
            if args.folder == "benchmark_molnet/molnet_random_pcba_c" or args.folder == "benchmark_molnet/molnet_random_muv_c":
                precision, recall, _ = precision_recall_curve([float(j) for j in y_test[i]], pre_pro)
                AUC = auc(recall, precision)
            aucs.append(AUC)
        return np.mean(aucs)

def get_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--folder', type=str, default="benchmark_molnet/molnet_random_bace_c")
    parser.add_argument('--hidden', type=int, default=512,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batch_size', type=float, default=128)
    parser.add_argument('--natom', type=float, default=50)
    parser.add_argument('--nclass', type=float, default=1)
    parser.add_argument('--type', type=str, default="classification")
    args = parser.parse_args()
    return args



def get_scores(gcn_scores,xgb_scores,names,args,device):
    for i in range(3):
        train_loss_list = []
        val_loss_list = []
        epoch_number = []
        args.dataset = args.folder + '/seed' + str(i + 1)
        load_data(args)
        feature_train, a_train, y_train = loadInputs_train(args)
        feature_val, a_val, y_val = loadInputs_val(args)
        feature_test, a_test, y_test = loadInputs_test(args)
        args.nclass = y_test.shape[-1]
        names['model' + str(i)] = GGNN(args.natom, args.hidden, args.nclass, args.dropout).to(device)
        optimizer = optim.Adam(names['model' + str(i)].parameters(), lr=args.lr)
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
                                                                criterion, args)
                val_loss, val_RMSE, val_R2, _,_ = evaluate(names['model' + str(i)], val_loader, criterion, args)
                print(f'\tTrain Loss: {train_loss:.3f}%')
                print(f'\t Val. RMSE: {val_RMSE:.3f}')
                writer.add_scalar('Train/Loss', train_loss, epoch)
                writer.add_scalar('val/Loss', val_loss, epoch)
                train_loss_list.append(train_loss)
                val_loss_list.append(val_loss)
                epoch_number.append(epoch)
                if epoch % 4 == 0 and epoch != 0:
                    decrease_learning_rate(optimizer, decrease_by=0.001)
            test_loss, test_RMSE, test_R2, _,MAE = evaluate(names['model' + str(i)], test_loader, criterion, args)
            print(f'\t test. RMSE: {test_RMSE:.3f}')
            train_dataset = ToxicDataset(feature_train, a_train, y_train)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
            xgb_train_feature = get_feature(names['model' + str(i)], train_loader, args)
            xgb_val_feature = get_feature(names['model' + str(i)], val_loader, args)
            xgb_test_feature = get_feature(names['model' + str(i)], test_loader, args)
            xgb_train_feature = xgb_train_feature.cpu().numpy()
            xgb_val_feature = xgb_val_feature.cpu().numpy()
            xgb_test_feature = xgb_test_feature.cpu().numpy()
            xgb_RMSE = xgb_regression(xgb_train_feature, y_train, xgb_val_feature, y_val, xgb_test_feature, y_test)
            if args.folder =="benchmark_molnet/molnet_random_qm7_r":
                gcn_scores.append([MAE])
            else:
                gcn_scores.append([test_RMSE])
            xgb_scores.append([xgb_RMSE])
        else:
            args.metric = 'AUC'
            criterion = nn.BCEWithLogitsLoss()
            for epoch in range(args.epochs):
                train_loss, val_total_loss = training_classing(names['model' + str(i)], train_loader, optimizer,
                                                               criterion, args)
                val_loss, val_AUC, val_precision, val_recall = evaluate_classion(names['model' + str(i)], val_loader,
                                                                                 criterion, args)
                print(f'\tTrain Loss: {train_loss:.3f}')
                print(f'\tVal AUC: {val_AUC:.3f}')
                print(f'\tVal. precision: {val_precision:.3f}')
                writer.add_scalar('Train/Loss', train_loss, epoch)
                writer.add_scalar('val/Loss', val_loss, epoch)
                train_loss_list.append(train_loss)
                val_loss_list.append(val_loss)
                epoch_number.append(epoch)
                if epoch % 4 == 0 and epoch != 0:
                    decrease_learning_rate(optimizer, decrease_by=0.001)
            test_loader_xgb = (torch.from_numpy(np.float32(feature_test)), torch.from_numpy(np.float32(a_test)),
                               torch.from_numpy(np.float32(y_test)))
            test_loss, test_AUC, test_precision, test_recall = evaluate_test_scros(names['model' + str(i)],
                                                                                   test_loader_xgb, criterion, args)
            train_dataset = ToxicDataset(feature_train, a_train, y_train)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
            xgb_train_feature = get_feature(names['model' + str(i)], train_loader, args)
            xgb_val_feature = get_feature(names['model' + str(i)], val_loader, args)
            xgb_test_feature = get_feature(names['model' + str(i)], test_loader, args)
            xgb_train_feature = xgb_train_feature.cpu().numpy()
            xgb_val_feature = xgb_val_feature.cpu().numpy()
            xgb_test_feature = xgb_test_feature.cpu().numpy()
            xgb_auc = xgboost_classion(xgb_train_feature, y_train, xgb_val_feature, y_val, xgb_test_feature, y_test)
            gcn_scores.append([test_AUC])
            xgb_scores.append([xgb_auc])

    return gcn_scores,xgb_scores

if __name__=="__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_argv()

    gcn_scores = []
    xgb_scores = []
    names = locals()
    gcn_scores,xgb_scores = get_scores(gcn_scores,xgb_scores,names,args,device)
    print(gcn_scores)
    print(xgb_scores)
    gcn_scores = np.array(gcn_scores)
    xgb_scores = np.array(xgb_scores)
    gcn_scores = np.nanmean(gcn_scores, axis=1)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(gcn_scores), np.nanstd(gcn_scores)
    print(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')

    xgb_scores = np.nanmean(xgb_scores, axis=1)  # average score for each model across tasks
    xgb_mean_score, xgb_std_score = np.nanmean(xgb_scores), np.nanstd(xgb_scores)
    print(f'Overall xgb_test {args.metric} = {xgb_mean_score:.6f} +/- {xgb_std_score:.6f}')





