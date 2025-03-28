import torch
import torch.nn as nn
from model.spd import SPDTransform, SPDTangentSpace, SPDRectified
import math
import torch.nn.functional as F

class signal2spd(nn.Module):
    # convert signal epoch to SPD matrix
    def __init__(self):
        super().__init__()
        self.dev = torch.device('cpu')
    def forward(self, x):
        
        x = x.squeeze()
        mean = x.mean(axis=-1).unsqueeze(-1).repeat(1, 1, x.shape[-1])
        x = x - mean
        cov = x@x.permute(0, 2, 1)
        cov = cov.to(self.dev)
        cov = cov/(x.shape[-1]-1)
        tra = cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        tra = tra.view(-1, 1, 1)
        cov /= tra
        identity = torch.eye(cov.shape[-1], cov.shape[-1], device=self.dev).to(self.dev).repeat(x.shape[0], 1, 1)
        cov = cov+(1e-5*identity)
        return cov 

class E2R(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.signal2spd = signal2spd()
    def patch_len(self, n, epochs):
        list_len=[]
        base = n//epochs
        for i in range(epochs):
            list_len.append(base)
        for i in range(n - base*epochs):
            list_len[i] += 1

        if sum(list_len) == n:
            return list_len
        else:
            return ValueError('check your epochs and axis should be split again')
    
    def forward(self, x):
        # x with shape[bs, ch, 1, time]
        list_patch = self.patch_len(x.shape[-1], int(self.epochs))
        x_list = list(torch.split(x, list_patch, dim=-1))
        for i, item in enumerate(x_list):
            x_list[i] = self.signal2spd(item)
        x = torch.stack(x_list).permute(1, 0, 2, 3)
        return x


class AttentionManifold(nn.Module):
    def __init__(self, in_embed_size, out_embed_size):
        super(AttentionManifold, self).__init__()
        
        self.d_in = in_embed_size
        self.d_out = out_embed_size
        self.q_trans = SPDTransform(self.d_in, self.d_out).cpu()
        self.k_trans = SPDTransform(self.d_in, self.d_out).cpu()
        self.v_trans = SPDTransform(self.d_in, self.d_out).cpu()

    def tensor_log(self, t):#4dim
        u, s, v = torch.svd(t)
        return u @ torch.diag_embed(torch.log(s)) @ v.permute(0, 1, 3, 2)
        
    def tensor_exp(self, t):#4dim
        # condition: t is symmetric!
        s, u = torch.linalg.eigh(t)
        return u @ torch.diag_embed(torch.exp(s)) @ u.permute(0, 1, 3, 2)
    
    def log_euclidean_distance(self, A, B):
        inner_term = self.tensor_log(A) - self.tensor_log(B)
        inner_multi = inner_term @ inner_term.permute(0, 1, 3, 2)
        _, s, _= torch.svd(inner_multi)
        final = torch.sum(s, dim=-1)
        return final

    def LogEuclideanMean(self, weight, cov):
        # cov:[bs, #p, s, s]
        # weight:[bs, #p, #p]
        bs = cov.shape[0]
        num_p = cov.shape[1]
        size = cov.shape[2]
        cov = self.tensor_log(cov).view(bs, num_p, -1)
        output = weight @ cov#[bs, #p, -1]
        output = output.view(bs, num_p, size, size)
        return self.tensor_exp(output)
        
    def forward(self, x, shape=None):
        if len(x.shape)==3 and shape is not None:
            x = x.view(shape[0], shape[1], self.d_in, self.d_in)
        x = x.to(torch.float)# patch:[b, #patch, c, c]
        q_list = []; k_list = []; v_list = []  
        # calculate Q K V
        bs = x.shape[0]
        m = x.shape[1]
        x = x.reshape(bs*m, self.d_in, self.d_in)
        Q = self.q_trans(x).view(bs, m, self.d_out, self.d_out)
        K = self.k_trans(x).view(bs, m, self.d_out, self.d_out)
        V = self.v_trans(x).view(bs, m, self.d_out, self.d_out)

        # calculate the attention score
        Q_expand = Q.repeat(1, V.shape[1], 1, 1)
    
        K_expand = K.unsqueeze(2).repeat(1, 1, V.shape[1], 1, 1 )
        K_expand = K_expand.view(K_expand.shape[0], K_expand.shape[1] * K_expand.shape[2], K_expand.shape[3], K_expand.shape[4])
        
        atten_energy = self.log_euclidean_distance(Q_expand, K_expand).view(V.shape[0], V.shape[1], V.shape[1])
        atten_prob = nn.Softmax(dim=-2)(1/(1+torch.log(1 + atten_energy))).permute(0, 2, 1)#now row is c.c.
        
        # calculate outputs(v_i') of attention module
        output = self.LogEuclideanMean(atten_prob, V)

        output = output.view(V.shape[0], V.shape[1], self.d_out, self.d_out)

        shape = list(output.shape[:2])
        shape.append(-1)

        output = output.contiguous().view(-1, self.d_out, self.d_out)
        return output, shape

class mAtt_param(nn.Module):
    def __init__(self, in_chan, s_chan, t_chan, downsample_rate, blocks, num_class):
        super().__init__()
        #FE
        self.meanpool = nn.AvgPool2d((1, downsample_rate))
        # bs, 1, channel, sample
        self.conv1 = nn.Conv2d(1, s_chan, (in_chan, 1))
        self.Bn1 = nn.BatchNorm2d(s_chan)
        # bs, 22, 1, sample
        self.conv2 = nn.Conv2d(s_chan, t_chan, (1, 12), padding=(0, 6))
        self.Bn2   = nn.BatchNorm2d(t_chan)
        
        # E2R
        self.ract1 = E2R(epochs=blocks)
        # riemannian part
        self.att2 = AttentionManifold(t_chan, t_chan)
        self.ract2  = SPDRectified()
        
        # R2E
        self.tangent = SPDTangentSpace(t_chan)
        self.flat = nn.Flatten()
        # fc
        self.linear = nn.Linear(t_chan * (t_chan + 1) // 2 * blocks, num_class, bias=True)

    def forward(self, x):                        # [128, 1, 22, 438]
        x = self.meanpool(x)
        x = self.conv1(x)                        # [128, 22, 1, 438]
        x = self.Bn1(x)                          # [128, 22, 1, 438]
        x = self.conv2(x)                        # [128, 20, 1, 439]
        x = self.Bn2(x)                          # [128, 20, 1, 439]
        
        x = self.ract1(x)                        # [128, 3, 20, 20]
        x, shape = self.att2(x)                  # [384, 18, 18]
        x = self.ract2(x)                        # [384, 18, 18]
        
        x = self.tangent(x)                      # [384, 171]
        x = x.view(shape[0], shape[1], -1)       # [128, 3, 171]
        fe = self.flat(x)                         # [128, 513]
        x = self.linear(fe)                       # [128, 4]
        return fe, x
    

class mAtt(nn.Module):
    def __init__(self, in_chan, downsample_rate, blocks, num_class):
        super().__init__()
        #FE
        self.meanpool = nn.AvgPool2d((1, downsample_rate))
        # bs, 1, channel, sample
        self.conv1 = nn.Conv2d(1, 22, (in_chan, 1))
        self.Bn1 = nn.BatchNorm2d(22)
        # bs, 22, 1, sample
        self.conv2 = nn.Conv2d(22, 20, (1, 12), padding=(0, 6))
        self.Bn2   = nn.BatchNorm2d(20)
        
        # E2R
        self.ract1 = E2R(epochs=blocks)
        # riemannian part
        self.att2 = AttentionManifold(20, 18)
        self.ract2  = SPDRectified()
        
        # R2E
        self.tangent = SPDTangentSpace(18)
        self.flat = nn.Flatten()
        # fc
        self.linear = nn.Linear(9*19*blocks, num_class, bias=True)

    def forward(self, x):  
        x = self.meanpool(x)                      # [128, 1, 22, 438]
        x = self.conv1(x)                        # [128, 22, 1, 438]
        x = self.Bn1(x)                          # [128, 22, 1, 438]
        x = self.conv2(x)                        # [128, 20, 1, 439]
        x = self.Bn2(x)                          # [128, 20, 1, 439]
        
        x = self.ract1(x)                        # [128, 3, 20, 20]
        x, shape = self.att2(x)                  # [384, 18, 18]
        x = self.ract2(x)                        # [384, 18, 18]
        
        x = self.tangent(x)                      # [384, 171]
        x = x.view(shape[0], shape[1], -1)       # [128, 3, 171]
        fe = self.flat(x)                         # [128, 513]
        x = self.linear(fe)                       # [128, 4]
        return fe, x