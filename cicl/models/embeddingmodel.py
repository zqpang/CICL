#from collections import defaultdict
#from typing import List, Optional
#from torch import Tensor
#from PIL import Image
#from einops import rearrange, repeat
#import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class Embedding_model(nn.Module):
    def __init__(self, d_model):
        super(Embedding_model,self).__init__()
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=2048, nhead=8, batch_first=True)

        self.num_features = d_model
        self.full_connect = nn.Linear(self.num_features * 2, self.num_features)
        init.kaiming_normal_(self.full_connect.weight, mode='fan_out')
        init.constant_(self.full_connect.bias, 0)
        
        self.full_bn = nn.BatchNorm1d(self.num_features)
        self.full_bn.bias.requires_grad_(False)
        init.kaiming_normal_(self.full_connect.weight, mode='fan_out')
        init.constant_(self.full_bn.bias, 0)

    def forward(self, features_img, features_black):
        
        features_stack = torch.stack((features_img.clone().detach(), features_black.clone().detach()), dim=1)
        features = self.encoder_layer(features_stack)
        features1, features2 = features.chunk(2,dim=1)
        features1 = features1.squeeze()
        features2 = features2.squeeze()
        features3 = torch.cat([features1,features2], dim = 1)
        features3 = self.full_connect(features3)
        features3 = self.full_bn(features3)
        bn_x = F.normalize(features3)
        
        
        if self.training == False:    
            return bn_x
        else:
            return bn_x
        

class Fusion_model(nn.Module):
    def __init__(self, d_model):
        super(Fusion_model, self).__init__()
        
        self.num_features = d_model
        
        #self.attention = nn.TransformerEncoderLayer(d_model=self.num_features, nhead=1, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=self.num_features, num_heads=1, batch_first=True)

        self.full_connect = nn.Linear(self.num_features * 2, self.num_features)
        init.kaiming_normal_(self.full_connect.weight, mode='fan_out')
        init.constant_(self.full_connect.bias, 0)
        
        self.full_bn = nn.BatchNorm1d(self.num_features)
        self.full_bn.bias.requires_grad_(False)
        init.kaiming_normal_(self.full_connect.weight, mode='fan_out')
        init.constant_(self.full_bn.bias, 0)

        # Feature transformation layers
        self.fc1 = nn.Linear(self.num_features, self.num_features)
        init.kaiming_normal_(self.fc1.weight, mode='fan_out')
        init.constant_(self.fc1.bias, 0)
        self.fc2 = nn.Linear(self.num_features, self.num_features)
        init.kaiming_normal_(self.fc2.weight, mode='fan_out')
        init.constant_(self.fc2.bias, 0)

    def forward(self, features_img, features_black):
        
        # Feature scaling
        #features_img = F.normalize(features_img)
        #features_black = F.normalize(features_black)
        
        features_black1 = F.relu(self.fc1(features_black))
        features_img1 = F.relu(self.fc2(features_img))
        hadamard_product = features_black1 * features_img1
        
        similarity_vector = F.softmax(hadamard_product, dim=-1)
        
        features_black = features_black + features_black * similarity_vector
        features_img = features_img + features_img * similarity_vector

        features_stack = torch.stack((features_img.clone().detach(), features_black.clone().detach()), dim=1)
        #features = self.encoder_layer(features_stack)
        features, _ = self.attention(features_stack, features_stack, features_stack)
        #print(features)
        features1, features2 = features.chunk(2, dim=1)
        features1 = features1.squeeze()
        features2 = features2.squeeze()
        
        # Feature transformation
        #features1 = F.relu(self.fc1(features1))
        #features2 = F.relu(self.fc2(features2))

        # Feature selection (example using max pooling)
        #features1 = torch.max(features1, dim=1)[0]

        #features2 = torch.max(features2, dim=1)[0]
        
        #print(features1.shape)


        features3 = torch.cat([features1, features2], dim=1)
        features3 = self.full_connect(features3)
        features3 = self.full_bn(features3)
        bn_x = F.normalize(features3)
        
        if self.training == False:    
            return bn_x
        else:
            return bn_x