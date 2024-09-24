from utils import _get_clones
import math
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict

class Router(nn.Module):
    def __init__(self, num_out_path, embed_size, hid):
        super(Router, self).__init__()
        self.num_out_path = num_out_path
        self.mlp = nn.Sequential(nn.Linear(embed_size*2, hid,bias=False),
                                nn.LayerNorm(normalized_shape = hid),
                                nn.ReLU(True), 
                                nn.Linear(hid, num_out_path, bias=False))

    def forward(self, x): # [B, clip_feature_dim]
        # x = x.mean(-2)#b,k,d
        # pdb.set_trace()
        x = self.mlp(x) # [B, 3]
        soft_g = torch.sigmoid(x)
        return soft_g

class SelfAttentionCell(nn.Module):
    def __init__(self, args):
        super(SelfAttentionCell, self).__init__()
        self.h = 8
        self.drop=0.0
        self.mlp_ratio = 4
        mlp_hidden_dim = int(args.embed_size * self.mlp_ratio)
        self.att_layer = AttentionLayer(args.embed_size, self.h, drop=self.drop)
        self.feed_forward_layer = FeedForward(args.embed_size, mlp_hidden_dim, drop=self.drop)
        self.dropout = nn.Dropout(self.drop)
        self.norm1 = nn.LayerNorm(args.embed_size)
        self.norm2 = nn.LayerNorm(args.embed_size)

    def forward(self, local_emb):
        mask=None 
        self_att_emb = self.dropout(self.att_layer(self.norm1(local_emb), mask=mask))
        out = self_att_emb + self.dropout(self.feed_forward_layer(self.norm2(self_att_emb)))
        return out

class AttentionLayer(nn.Module):
    def __init__(self, embed_size, h, is_share=False, drop=0.0):
        super(AttentionLayer, self).__init__()
        self.is_share = is_share
        self.h = h
        self.embed_size = embed_size
        self.d_k = embed_size // h
        self.drop_p = drop
        if is_share:
            self.linear = nn.Linear(embed_size, embed_size)
            self.linears = [self.linear, self.linear, self.linear] 
        else:
            self.linears = _get_clones(nn.Linear(embed_size, embed_size), 3)
        if self.drop_p > 0:
            self.dropout = nn.Dropout(drop)
        
    def forward(self, inp, mask=None):
        nbatches = inp.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (inp, inp, inp))]
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)     
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.drop_p > 0:
            p_attn = self.dropout(p_attn)
        x = torch.matmul(p_attn, value) 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return x
    

class FeedForward(nn.Module):
    def __init__(self, embed_size, hidden, drop=0.0):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, hidden)
        self.fc2 = nn.Linear(hidden, embed_size)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))
    
    
class CrossAttention(nn.Module):
    def __init__(self, clip_feature_dim, n_layers, n_heads, attn_mask=None):
        super(CrossAttention, self).__init__()
        self.n_layers = n_layers
        self.resblocks = _get_clones(ResidualCrossAttentionBlock(clip_feature_dim, n_heads, attn_mask), n_layers)

    def forward(self, x, y):
        for i in range(self.n_layers):
            x = self.resblocks[i](x, y)
        return x
    
class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None):
        super(ResidualCrossAttentionBlock, self).__init__()

        self.attn = CrossAttentionLayer(d_model, n_head)
        self.ln_x1 = nn.LayerNorm(d_model)
        self.ln_y1 = nn.LayerNorm(d_model)
        self.mlp_ratio = 4
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, int(d_model * self.mlp_ratio))),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(int(d_model * self.mlp_ratio), d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x, y):
        return self.attn(x, y)
        
    def forward(self, x, y):
        x = x + self.attention(self.ln_x1(x), self.ln_y1(y))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, n_head):
        super(CrossAttentionLayer, self).__init__()
        self.h = n_head
        self.d_model = d_model
        self.d_k = d_model // n_head
        self.projections = _get_clones(nn.Linear(d_model, d_model), 3)

    def forward(self, x, y):
        nbatches = x.size(0)
        query, key, value = [l(v).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                            for l, v in zip(self.projections, (y, x, x))]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        p_attn = F.softmax(scores, dim=-1)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return x
    
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
    