import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
#

class graphconvolution(nn.Module):
    def __init__(self,input_features,output_features):
        super(graphconvolution, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = Parameter(torch.FloatTensor(input_features, output_features), requires_grad = True)
        # self.weight = Variable(torch.FloatTensor(input_features,output_features),requires_grad=True).cuda()
        self.bias = Parameter(torch.FloatTensor(output_features), requires_grad = True)
        self.fc = nn.Linear(input_features,output_features)
    def forward(self,feature,A):
        # feature_w = torch.matmul(feature,self.weight)+self.bias
        feature_w = F.relu(self.fc(feature))
        feature_a = torch.bmm(A,feature_w)
        # feature_a = feature_a+feature_w
        return feature_a

class GCN(nn.Module):
    def __init__(self,nfeat,nhid,nclass,dropout):
        super(GCN, self).__init__()
        self.gc1 = graphconvolution(nfeat,nhid)
        self.gc2 = graphconvolution(nhid,nhid)
        self.gc3 = graphconvolution(nhid,nhid)
        self.dropout = dropout
        self.fc = nn.Linear(nhid,nclass)
    def forward(self,feature,A):
        feature_gc = self.gc1(feature,A)
        for i in range(5):
            feature_gc = self.gc2(feature_gc, A)
        feature_gc = self.gc3(feature_gc,A)
        feature_sum_hidden = torch.sum(feature_gc,dim=1)
        feature_sum = self.fc(feature_sum_hidden)
        return feature_sum,feature_sum_hidden

