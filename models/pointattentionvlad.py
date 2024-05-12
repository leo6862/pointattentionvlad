import torch
import torch.nn as nn
import numpy as np
import sys 
import math
from torch.autograd import Variable
from utils.pointnet_util import *
import torch.nn.functional as F


def idx_points(points, idx):
    """

    Input:
        points: input points data, [B, N,nn_num, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx,:, :]
    return new_points


def group_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)

class NetVLADLoupe(nn.Module):
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True, is_training=True):
        super(NetVLADLoupe, self).__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.is_training = is_training
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.softmax = nn.Softmax(dim=-1)

        self.cluster_weights = nn.Parameter(torch.randn(
            feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.cluster_weights2 = nn.Parameter(torch.randn(
            1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.hidden1_weights = nn.Parameter(torch.randn(
            cluster_size * feature_size, output_dim) * 1 / math.sqrt(feature_size))

        if add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(
                cluster_size) * 1 / math.sqrt(feature_size))
            self.bn1 = None

        self.bn2 = nn.BatchNorm1d(output_dim)

        if gating:
            self.context_gating = GatingContext(
                output_dim, add_batch_norm=add_batch_norm)

    def forward(self, x):
        x = x.transpose(1, 3).contiguous()
        x = x.view((-1, self.max_samples, self.feature_size))
        activation = torch.matmul(x, self.cluster_weights)
        if self.add_batch_norm:
            # activation = activation.transpose(1,2).contiguous()
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)
            activation = activation.view(-1, self.max_samples, self.cluster_size)
        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation)
        activation = activation.view((-1, self.max_samples, self.cluster_size))

        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2

        activation = torch.transpose(activation, 2, 1)
        x = x.view((-1, self.max_samples, self.feature_size))
        vlad = torch.matmul(activation, x)
        vlad = torch.transpose(vlad, 2, 1)
        vlad = vlad - a

        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = vlad.reshape((-1, self.cluster_size * self.feature_size))
        vlad = F.normalize(vlad, dim=1, p=2)

        vlad = torch.matmul(vlad, self.hidden1_weights)

        vlad = self.bn2(vlad)

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad

class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(
            torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)

        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)

        activation = x * gates

        return activation

class Local_Attention(nn.Module):
    def __init__(self,input_dim,d_model):
        super(Local_Attention,self).__init__()
        self.w_q = nn.Linear(input_dim,d_model)
        self.w_k = nn.Linear(input_dim,d_model)
        self.w_v = nn.Linear(input_dim,d_model)
        
        self.fc_pos_encoding = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_attn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc2 = nn.Linear(d_model, input_dim)
    def forward(self,x,xyz):
        #input x.shape = [B N' nn_num hidden_channel_2]
        #xyz.shape = [B N' nn_num 3]
        raw_feature = x[:,:,0,:] #B N' 1 hidden_channel_2
        raw_xyz = xyz[:,:,0,:].unsqueeze(-2)
        q = self.w_q(raw_feature)
        k = self.w_k(x)
        v = self.w_v(x)
        pos_enc = self.fc_pos_encoding(xyz - raw_xyz)
        
        attn =  self.fc_attn(q.unsqueeze(-2) - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f 

        res = torch.einsum('bmnf,bmnf->bmf',attn,v + pos_enc) 
        res = self.fc2(res) + x[:,:,0,:] 
        return res 
def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)

class PointAttentionVLAD(nn.Module):
    def __init__(self,num_points=4096,output_dim=256,layer_number=3):
        super(PointAttentionVLAD,self).__init__()
        inchannel = 3
        hidden_channel_1 = 64
        hidden_channel_2 = 256
        hidden_channel_3 = 1024
        self.conv1 = nn.Conv1d(inchannel,hidden_channel_1,1)
        self.conv2 = nn.Conv1d(hidden_channel_1,hidden_channel_2,1)
        self.conv3 = nn.Conv1d(hidden_channel_2*2,hidden_channel_3,1)
        self.bn1 = nn.BatchNorm1d(hidden_channel_1)
        self.bn2 = nn.BatchNorm1d(hidden_channel_2)
        self.bn3 = nn.BatchNorm1d(hidden_channel_3)
        self.local_attention = Local_Attention(hidden_channel_2,440)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_channel_2, nhead=2, dim_feedforward=512, activation='relu')
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)


        self.net_vlad = NetVLADLoupe(feature_size=hidden_channel_3, max_samples=2000, cluster_size=64,
                                     output_dim=output_dim, gating=True, add_batch_norm=True,
                                     is_training=True)
    def forward(self,x):
        """
        input shape :  B 1 N 19
        """

        batch_size = x.shape[0]
        nn_index = x[:,:,:,3:].to(torch.int64) #B 1 N 16
        x = x[:,:,:,:3] # B 1 N 3
        nn_index = nn_index.reshape(nn_index.shape[0],nn_index.shape[2],-1) #B N 16

        x_raw = x.reshape(batch_size,x.shape[-2],x.shape[-1]) # B N 3

        #将原始的 点云[B N 3] -> [B 3 N] -conv1d> [B hidden_channel_1 N] -conv1d> [B hidden_channel_2 N]
        x = F.relu(self.bn1(self.conv1(x_raw.permute(0,2,1))))
        x = F.relu(self.bn2(self.conv2(x)))
        
        #fps & idx_points [B hidden_channel_2 k N']
        fps_idx = farthest_point_sample(x_raw, 2000) #B 2000
        grouped_x = group_points(x.permute(0,2,1),nn_index) # B N 16 hidden_channel_2
        grouped_x = grouped_x - x.permute(0,2,1).unsqueeze(-2) #B N nn_num hidden_channel_2
        grouped_x = idx_points(grouped_x,fps_idx) # B N' nn_num hidden_channel_2
        
        grouped_raw_xyz = group_points(x_raw,nn_index)
        grouped_xyz = idx_points(grouped_raw_xyz,fps_idx) #[B N' nn_num 3]
        
        #local attention [B N' hidden_channel_2]
        grouped_x = self.local_attention(grouped_x,grouped_xyz)
        
        
        
        global_attention_grouped_x = self.transformer_encoder(grouped_x.permute(1,0,2)) 
        global_attention_grouped_x = torch.cat((global_attention_grouped_x.permute(1,2,0),grouped_x.permute(0,2,1)),dim=1)#permute # B C N
        
        
        #[B 2*hidden_channel_2 N']  conv1d -> [B 1024 N']
        x = F.relu(self.bn3(self.conv3(global_attention_grouped_x))) 
        
        
        x = x.reshape(x.shape[0],x.shape[1],x.shape[2],1) # B C N 1
        x = self.net_vlad(x)
        torch.cuda.empty_cache()
        print('max mem : {}'.format(torch.cuda.max_memory_allocated()/1024/1024/1024))
        return x
if __name__ == "__main__":
    import thop
    model = PointAttentionVLAD().cpu()
    data_xyz = torch.randn((30,1,4096,3)).cpu()
    data_index = torch.randint(0,4096,[30,1,4096,16]).cpu()
    data = torch.cat((data_xyz,data_index),dim=-1)
    out = model(data.cpu())
    print('out.shape = {}'.format(out.shape))
    model_structure(model)
    flops, params = thop.profile(model, inputs=(data,),custom_ops = {})
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')


