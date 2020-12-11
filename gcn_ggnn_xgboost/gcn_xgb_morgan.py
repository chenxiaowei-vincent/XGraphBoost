
import numpy as np
import torch
import argparse
import os
from tensorboardX import SummaryWriter
from gcn_model.gcn_xgboost_scores import get_scores

writer = SummaryWriter('loss')


def get_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200,#800
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
    parser.add_argument('--nclass', type=float, default=1)
    parser.add_argument('--type', type=str, default="classification")
    args = parser.parse_args()
    return args




if __name__=="__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = get_argv()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gcn_scores = []
    xgb_gcn = []
    gcn_morgan = []
    morgan_scores = []
    gcn_scores,xgb_gcn,gcn_morgan,morgan_scores = get_scores(gcn_scores,xgb_gcn,gcn_morgan,morgan_scores,args)
    writer.close()
    gcn_scores,xgb_gcn,gcn_morgan,morgan_scores = np.array(gcn_scores),np.array(xgb_gcn),np.array(gcn_morgan),np.array(morgan_scores)

    gcn_scores = np.nanmean(gcn_scores, axis=1)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(gcn_scores), np.nanstd(gcn_scores)
    print(f'Overall gcn test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')

    xgb_gcn = np.nanmean(xgb_gcn, axis=1)  # average score for each model across tasks
    xgb_mean_score, xgb_std_score = np.nanmean(xgb_gcn), np.nanstd(xgb_gcn)
    print(f'Overall xgb_gcn_auc {args.metric} = {xgb_mean_score:.6f} +/- {xgb_std_score:.6f}')

    morgan_scores = np.nanmean(morgan_scores, axis=1)  # average score for each model across tasks
    morgan_mean_score, morgan_std_score = np.nanmean(morgan_scores), np.nanstd(morgan_scores)
    print(f'Overall morgan_test {args.metric} = {morgan_mean_score:.6f} +/- {morgan_std_score:.6f}')

    gcn_morgan = np.nanmean(gcn_morgan, axis=1)  # average score for each model across tasks
    gcn_morgan_mean_score, gcn_morgan_std_score = np.nanmean(gcn_morgan), np.nanstd(gcn_morgan)
    print(f'Overall gcn_morgan_auc {args.metric} = {gcn_morgan_mean_score:.6f} +/- {gcn_morgan_std_score:.6f}')

