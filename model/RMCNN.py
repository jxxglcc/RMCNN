import torch
import torch.nn as nn
from model.spd import SPDTransform, SPDTangentSpace, SPDRectified

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
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks
        self.signal2spd = signal2spd()
    def patch_len(self, n, blocks):
        list_len=[]
        base = n//blocks
        for i in range(blocks):
            list_len.append(base)
        for i in range(n - base*blocks):
            list_len[i] += 1

        if sum(list_len) == n:
            return list_len
        else:
            return ValueError('check your blocks and axis should be split again')
    
    def forward(self, x):
        # x with shape[bs, ch, 1, time]
        list_patch = self.patch_len(x.shape[-1], int(self.blocks))
        x_list = list(torch.split(x, list_patch, dim=-1))
        for i, item in enumerate(x_list):
            x_list[i] = self.signal2spd(item)
        x = torch.stack(x_list).permute(1, 0, 2, 3)
        return x


class e_mean(nn.Module):
    def __init__(self, in_embed_size, out_embed_size):
        super(e_mean, self).__init__()
        
        self.d_in = in_embed_size
        self.d_out = out_embed_size
        self.time_bimap = SPDTransform(self.d_in, self.d_out).cpu()
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

    def LogEuclideanMean(self, cov):
        # cov:[bs, #p, s, s]
        # weight:[bs, #p, #p]
        bs = cov.shape[0]
        num_p = cov.shape[1]
        size = cov.shape[2]
        cov = self.tensor_log(cov).view(bs, num_p, -1)
        output = torch.mean(cov, dim=1).view(bs, size, size)#[bs, size, size]
        output = output.unsqueeze(1)
        return self.tensor_exp(output).squeeze(1)
        
    def forward(self, x):
        bs = x.shape[0]
        m = x.shape[1]
        x = x.reshape(bs*m, self.d_in, self.d_in)
        x = self.time_bimap(x).view(bs, m, self.d_out, self.d_out)

        # x = self.LogEuclideanMean(x)
        x = torch.mean(x, dim=1)
        # x = x.reshape(bs, fb, self.d_out, self.d_out)
        return x

class RMCNN(nn.Module):
    def __init__(self, in_chan, st_chan, r_chan, fb, downsample_rate, blocks, num_class):
        super().__init__()

        self.freq_bands = len(fb)
        self.blocks = blocks
        #FE
        self.meanpool = nn.AvgPool2d((1, downsample_rate))

        self.conv = nn.Conv2d(1, st_chan, (in_chan, 13), padding=(0, 6))
        self.bn   = nn.BatchNorm2d(st_chan)
        
        # E2R
        self.e2r = E2R(blocks=blocks)
        # riemannian part
        self.bimap = e_mean(st_chan, r_chan)
        self.rec  = SPDRectified()
        # R2E
        self.tangent = SPDTangentSpace(r_chan)
        self.flat = nn.Flatten()
        if len(fb) > 2:
            self.linear = nn.Linear(self.freq_bands * r_chan * (r_chan + 1) // 2, num_class, bias=True)
        else:
            self.linear = nn.Linear(r_chan * (r_chan + 1) // 2, num_class, bias=True)

    def forward(self, x): 
        bs = x.shape[0]
        if self.freq_bands > 2:
            x = x.reshape(bs*self.freq_bands, 1, x.shape[2], x.shape[3])
        x = self.meanpool(x)
        x = self.conv(x)
        x = self.bn(x)
        
        x = self.e2r(x)
        x = self.bimap(x)
        c = self.rec(x)
        fe = self.tangent(c)
        if self.freq_bands > 2:
            fe = fe.reshape(bs, self.freq_bands, fe.shape[1]) 
            fe = self.flat(fe)
        x = self.linear(fe)
        return fe, x