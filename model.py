import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np

class EncoderLayer(nn.Module):
    '''
        Transformer Encoder, which will be used for both context interpreter and aggregator.
    '''
    def __init__(self, n_head, n_hid, att_dropout = 0.5, ffn_dropout = 0.2, res_dropout = 0.2):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(n_head, n_hid, att_dropout)
        self.feed_forward = PositionwiseFeedForward(n_hid, ffn_dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(n_hid, res_dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

    
class PositionalEncoding(nn.Module):
    '''
        Implement the Position Encoding (Sinusoid) function.
    '''
    def __init__(self, n_hid, max_len = 1000, dropout = 0.2):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_hid)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = 1 / (10000 ** (torch.arange(0., n_hid, 2.)) / n_hid)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(0) / np.sqrt(n_hid)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.dropout(x + Variable(self.pe[:, :, :x.shape[-2]], requires_grad=False))    

class LayerNorm(nn.Module):
    '''
        Construct a layernorm module.
    '''
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std  = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, n_hid, dropout=0.3):
        '''
            Multihead self-attention that can calcualte mutual attention table
            based on which to aggregate embedding at different position.
        '''
        super(MultiHeadedAttention, self).__init__()
        self.d_k = n_hid // n_head
        self.h = n_head
        self.linears = nn.ModuleList([nn.Linear(n_hid, n_hid) for _ in range(3)])
        self.out = nn.Linear(self.d_k * n_head, n_hid)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from n_hid => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = self.attention(query, key, value, mask=mask, 
                                dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.out(x)

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    
class PositionwiseFeedForward(nn.Module):
    '''
        Implements FFN equation (1-D convolution).
    '''
    def __init__(self, n_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(n_hid, n_hid * 2)
        self.w_2 = nn.Linear(n_hid * 2, n_hid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
        
   
class SublayerConnection(nn.Module):
    '''
        A residual connection followed by a layer norm.
    '''
    def __init__(self, size, dropout = 0.3):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return self.norm(x + self.dropout(sublayer(x)))
        
class PositionalAttention(nn.Module):
    '''
        A simple positional attention layer that assigns different weights for word in different relative position.
    '''
    def __init__(self, n_seq):
        super(PositionalAttention, self).__init__()
        self.pos_att = nn.Parameter(torch.ones(n_seq))
    def forward(self, x):
        # x: L * d -> d * L
        return (x.transpose(-2, -1) * self.pos_att).transpose(-2, -1)

class Model(nn.Module):
    def __init__(self, n_head, n_hid, n_seq, n_layer, item2vec):
        super(Model, self).__init__()
        self.n_hid = n_hid
        self.n_seq = n_seq
        self.n_layer = n_layer
        self.emb = nn.Embedding(len(item2vec), n_hid)
        self.update_embedding(item2vec)

        self.context_encoder = nn.ModuleList([EncoderLayer(n_head, n_hid) for _ in range(n_layer)]) 
        self.context_aggegator = nn.ModuleList([EncoderLayer(n_head, n_hid) for _ in range(n_layer)])
        self.pos_att = PositionalAttention(n_seq)
        self.pos_enc = PositionalEncoding(n_hid)
        self.out = nn.Linear(n_hid, n_hid) # Can consider tie weights with input embedding.

    def update_embedding(self, item2vec, init = False):
        target_item2vec = torch.FloatTensor(item2vec)
        self.emb.weight = nn.Parameter(target_item2vec)
        self.emb.weight.requires_grad = False
            
    def mask_pad(self, x, pad = 0):
        "Create a mask to hide padding"
        return (x != pad).unsqueeze(-2).unsqueeze(-2)

    def forward(self, contexts, pad = 0):
        #  B (Batch Size) * K (number of contexts) * L (Length of each context) -> K * [B * L] ->
        masks = self.mask_pad(contexts, pad).transpose(0,1)
        x = self.pos_enc(self.pos_att(self.emb(contexts))).transpose(0,1)
        res = []
        for xi, mask in zip(x, masks):
            for layer in self.context_encoder:
                xi = layer(xi, mask)                
                
            mask_value = mask.squeeze(1).squeeze(1).unsqueeze(-1).float()
            res += [torch.sum(xi * mask_value, dim=1) / torch.sum(mask_value, dim=1)]
        #  K * B * n_hid  -> B * K * n_hid
        res = torch.stack(res).transpose(0,1)            
            
        for layer in self.context_aggegator:
            res = layer(res, None)
        res = res.mean(dim=1)
        return self.out(res)
