import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from load_data_batch import loadInputs_feature_smiles,ToxicDataset,load_data,get_morgan_feature
from torch.utils.data import Dataset
from gcn_model.gcn_xgboost import xgb_regression,xgboost_classion
from gcn_model.gcn_model import GCN
import pandas as pd
from tensorboardX import SummaryWriter
from data_deal import decrease_learning_rate
from gcn_model.gcn_training import training,evaluate,training_classing,evaluate_classion,evaluate_test_scros

writer = SummaryWriter('loss')


def get_feature(model,data, device):
    model.eval()
    i = 0
    with torch.no_grad():
        for k in data:
            feature, A, y = k
            if len(y.shape) == 3:
                y = y.squeeze(1)
            feature, A, y = feature.to(device), A.to(device), y.to(device)
            output, feature = model(feature, A)
            if i ==0:
                features = feature
            else:
                features = torch.cat((features,feature))
            i = i+1
    return features


def get_scores(gcn_scores,xgb_gcn,gcn_morgan,morgan_scores,args):
    names = locals()
    for i in range(3):
        train_loss_list = []
        val_loss_list = []
        epoch_number = []
        args.dataset = args.folder + '/seed' + str(i+1)
        load_data(args)
        feature_train,a_train,y_train,smiles_train = loadInputs_feature_smiles(args,'train')
        feature_val,a_val,y_val,smiles_val = loadInputs_feature_smiles(args,'val')
        feature_test,a_test,y_test,smiles_test = loadInputs_feature_smiles(args,'test')
        args.nclass = y_test.shape[-1]
        names['model' + str(i) ] = GCN(args.natom, args.hidden, args.nclass, args.dropout).to(args.device)
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
                train_loss,train_R2,val_total_loss = training(names['model' + str(i) ], train_loader, optimizer, criterion, args.device)
                val_loss,val_RMSE,val_R2,_,_ = evaluate(names['model' + str(i) ], val_loader, criterion, args.device)
                print(f'\tTrain Loss: {train_loss:.3f}%')
                print(f'\t Val. RMSE: {val_RMSE:.3f}')
                writer.add_scalar('Train/Loss', train_loss, epoch)
                writer.add_scalar('val/Loss', val_loss, epoch)
                train_loss_list.append(train_loss)
                val_loss_list.append(val_loss)
                epoch_number.append(epoch)
                if epoch % 4 == 0 and epoch != 0:
                    decrease_learning_rate(optimizer, decrease_by=0.001)
            test_loss, test_RMSE, test_R2, _,MAE = evaluate(names['model' + str(i) ], test_loader, criterion, args.device)
            print(f'\t test. RMSE: {test_RMSE:.3f}')
            train_loader_xgb = (torch.from_numpy(np.float32(feature_train)), torch.from_numpy(np.float32(a_train)),
                            torch.from_numpy(np.float32(y_train)))
            val_loader_xgb = (torch.from_numpy(np.float32(feature_val)), torch.from_numpy(np.float32(a_val)),
                          torch.from_numpy(np.float32(y_val)))
            test_loader_xgb = (torch.from_numpy(np.float32(feature_test)), torch.from_numpy(np.float32(a_test)),
                           torch.from_numpy(np.float32(y_test)))
            xgb_train_feature = get_feature(names['model' + str(i) ], train_loader_xgb, args.device)
            xgb_val_feature = get_feature(names['model' + str(i) ], val_loader_xgb, args.device)
            xgb_test_feature = get_feature(names['model' + str(i) ], test_loader_xgb, args.device)
            xgb_train_feature = xgb_train_feature.cpu().numpy()
            xgb_val_feature = xgb_val_feature.cpu().numpy()
            xgb_test_feature = xgb_test_feature.cpu().numpy()
            xgb_gcn_rmse = xgb_regression(xgb_train_feature, y_train, xgb_val_feature, y_val, xgb_test_feature, y_test,args)
            morgan_train_feature,morgan_val_feature,morgan_test_feature = get_morgan_feature(smiles_train),get_morgan_feature(smiles_val),get_morgan_feature(smiles_test)
            gcn_morgan_train_feature = pd.concat([pd.DataFrame(xgb_train_feature),morgan_train_feature],axis=1)
            gcn_morgan_val_feature = pd.concat([pd.DataFrame(xgb_val_feature),morgan_val_feature],axis=1)
            gcn_morgan_test_feature = pd.concat([pd.DataFrame(xgb_test_feature),morgan_test_feature],axis=1)
            gcn_morgan_train_feature.columns = gcn_morgan_val_feature.columns = gcn_morgan_test_feature.columns = range(gcn_morgan_test_feature.shape[1])
            morgan_rmse= xgb_regression(morgan_train_feature, y_train, morgan_val_feature, y_val, morgan_test_feature, y_test,args)
            gcn_morgan_rmse = xgb_regression(gcn_morgan_train_feature, y_train, gcn_morgan_val_feature, y_val, gcn_morgan_test_feature, y_test,args)
            xgb_RMSE = xgb_regression(xgb_train_feature, y_train, xgb_val_feature, y_val, xgb_test_feature, y_test,args)
            gcn_scores.append([test_RMSE])
            xgb_gcn.append([xgb_RMSE])
            gcn_morgan.append([gcn_morgan_rmse])
            morgan_scores.append([morgan_rmse])
        else:
            args.metric = 'AUC'
            criterion = nn.BCEWithLogitsLoss()
            for epoch in range(args.epochs):
                train_loss,val_total_loss = training_classing(names['model' + str(i) ], train_loader, optimizer, criterion, args.device)
                val_loss,val_AUC,val_precision,val_recall = evaluate_classion(names['model' + str(i) ], val_loader, criterion, args.device)
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
            train_loader_xgb = (torch.from_numpy(np.float32(feature_train)), torch.from_numpy(np.float32(a_train)),
                            torch.from_numpy(np.float32(y_train)))
            val_loader_xgb = (torch.from_numpy(np.float32(feature_val)), torch.from_numpy(np.float32(a_val)),
                          torch.from_numpy(np.float32(y_val)))
            test_loader_xgb = (torch.from_numpy(np.float32(feature_test)), torch.from_numpy(np.float32(a_test)),
                           torch.from_numpy(np.float32(y_test)))
            test_loss, test_AUC, test_precision, test_recall = evaluate_test_scros(names['model' + str(i) ], test_loader_xgb, criterion, args.device)
            xgb_train_feature = get_feature(names['model' + str(i) ], train_loader_xgb, args.device)
            xgb_val_feature = get_feature(names['model' + str(i) ], val_loader_xgb, args.device)
            xgb_test_feature = get_feature(names['model' + str(i) ], test_loader_xgb, args.device)
            xgb_train_feature = xgb_train_feature.cpu().numpy()
            xgb_val_feature = xgb_val_feature.cpu().numpy()
            xgb_test_feature = xgb_test_feature.cpu().numpy()
            xgb_gcn_one_auc = xgboost_classion(xgb_train_feature, y_train, xgb_val_feature, y_val, xgb_test_feature, y_test,args)
            morgan_train_feature,morgan_val_feature,morgan_test_feature = get_morgan_feature(smiles_train),get_morgan_feature(smiles_val),get_morgan_feature(smiles_test)
            gcn_morgan_train_feature = pd.concat([pd.DataFrame(xgb_train_feature),morgan_train_feature],axis=1)
            gcn_morgan_val_feature = pd.concat([pd.DataFrame(xgb_val_feature),morgan_val_feature],axis=1)
            gcn_morgan_test_feature = pd.concat([pd.DataFrame(xgb_test_feature),morgan_test_feature],axis=1)
            gcn_morgan_train_feature.columns = gcn_morgan_val_feature.columns = gcn_morgan_test_feature.columns = range(gcn_morgan_test_feature.shape[1])
            morgan_auc= xgboost_classion(morgan_train_feature, y_train, morgan_val_feature, y_val, morgan_test_feature, y_test,args)
            gcn_morgan_one_auc = xgboost_classion(gcn_morgan_train_feature, y_train, gcn_morgan_val_feature, y_val, gcn_morgan_test_feature, y_test,args)
            gcn_scores.append([test_AUC])
            xgb_gcn.append([xgb_gcn_one_auc])
            gcn_morgan.append([gcn_morgan_one_auc])
            morgan_scores.append([morgan_auc])
    return gcn_scores,xgb_gcn,gcn_morgan,morgan_scores