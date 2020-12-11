
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, precision_score, recall_score,r2_score,mean_absolute_error,mean_squared_error
import pandas as pd


def training(model, data,optimizer, criterion, device):
    model.train()
    total_loss = []
    for k in data:
        feature,A,y = k
        if len(y.shape) == 3:
            y = y.squeeze(1)
        model.zero_grad()
        feature,A, y = feature.to(device),A.to(device), y.to(device)
        output,_ = model(feature,A)
        loss = criterion(output, y)
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
            feature, A, y = feature.to(device), A.to(device), y.to(device)
            output,_ = model(feature,A)
            valid_loss = criterion(output, y)
            val_total_loss = val_total_loss+valid_loss
    return (sum(total_loss) / len(total_loss)),R2,val_total_loss

def evaluate(model, data, criterion, device):
    model.eval()
    total_loss = []
    with torch.no_grad():
        for k in data:
            feature, A, y = k
            if len(y.shape) == 3:
                y = y.squeeze(1)
            feature, A, y = feature.to(device), A.to(device), y.to(device)
            output,feature = model(feature,A)
            total_loss.append((criterion(output,  y)).item())
            MSE = mean_squared_error(y.cpu().numpy(), output.cpu().numpy())
            RMSE = MSE ** 0.5
            R2 = r2_score(y.cpu().numpy(), output.cpu().numpy())
            mae = mean_absolute_error(y.cpu().numpy(), output.cpu().numpy())
    return (sum(total_loss) / len(total_loss)),RMSE,R2,feature,mae


def training_classing(model, data,optimizer, criterion, device):
    model.train()
    total_loss = []
    for k in data:
        feature,A,y = k
        if len(y.shape)==3:
            y = y.squeeze(1)
        model.zero_grad()
        feature,A, y = feature.to(device),A.to(device), y.to(device)
        output,_ = model(feature,A)
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
            feature, A, y = feature.to(device), A.to(device), y.to(device)
            output,_ = model(feature,A)
            valid_loss = criterion(output, y)
            val_total_loss = val_total_loss+valid_loss
    return (sum(total_loss) / len(total_loss)),val_total_loss

def evaluate_classion(model, data, criterion, device):
    model.eval()
    total_loss = []
    y_predict = []
    y_test = []
    with torch.no_grad():
        for k in data:
            feature, A, y = k
            if len(y.shape) == 3:
                y = y.squeeze(1)
            feature, A, y = feature.to(device), A.to(device), y.to(device)
            output,_ = model(feature,A)
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


def evaluate_test_scros(model, data, criterion, device):
    model.eval()
    total_loss = []
    y_predict = []
    y_test = []
    with torch.no_grad():
        feature, A, y = data
        if len(y.shape) == 3:
            y = y.squeeze(1)
        feature, A, y = feature.to(device), A.to(device), y.to(device)
        output,_ = model(feature,A)
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
            AUC_all.append(AUC)
            precision_all.append(precision)
            recall_all.append(recall)
        AUC = np.mean(AUC_all)
        precision = np.mean(precision_all)
        recall = np.mean(recall_all)

    return (sum(total_loss) / len(total_loss)),AUC,precision,recall
