import torch
from torch import nn
import math
from utils import _get_clones
from collections import OrderedDict
import torch.nn.functional as F

class BertBiAttention(nn.Module):
    def __init__(self,num_attention_heads=8,hidden_size=512,dropout_rate=0.1):
        super(BertBiAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads) )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads) #每个头的维度
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query1 = nn.Linear(hidden_size, self.all_head_size)
        self.key1 = nn.Linear(hidden_size, self.all_head_size)
        self.value1 = nn.Linear(hidden_size, self.all_head_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.query2 = nn.Linear(hidden_size, self.all_head_size)
        self.key2 = nn.Linear(hidden_size, self.all_head_size)
        self.value2 = nn.Linear(hidden_size, self.all_head_size)
        self.dropout2 = nn.Dropout(dropout_rate)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        input_tensor1,
        input_tensor2,
    ):

        mixed_query_layer1 = self.query1(input_tensor1)
        mixed_key_layer1 = self.key1(input_tensor1)
        mixed_value_layer1 = self.value1(input_tensor1)

        query_layer1 = self.transpose_for_scores(mixed_query_layer1)
        key_layer1 = self.transpose_for_scores(mixed_key_layer1)
        value_layer1 = self.transpose_for_scores(mixed_value_layer1)
        mixed_query_layer2 = self.query2(input_tensor2)
        mixed_key_layer2 = self.key2(input_tensor2)
        mixed_value_layer2 = self.value2(input_tensor2)

        query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)

        attention_scores1 = torch.matmul(query_layer2, key_layer1.transpose(-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
        # attention_scores1 = attention_scores1 * attention_mask1

        attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)
        attention_probs1 = self.dropout1(attention_probs1)

        context_layer1 = torch.matmul(attention_probs1, value_layer1)
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
        context_layer1 = context_layer1.view(*new_context_layer_shape1)

        attention_scores2 = torch.matmul(query_layer1, key_layer2.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)

        # attention_scores2 = attention_scores2 * attention_mask2
        attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)

        attention_probs2 = self.dropout2(attention_probs2)
        context_layer2 = torch.matmul(attention_probs2, value_layer2)
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape2 = context_layer2.size()[:-2] + (self.all_head_size,)
        context_layer2 = context_layer2.view(*new_context_layer_shape2)

        attn_data = {
                "attn2": attention_probs2,
                "queries2": query_layer2,
                "keys1": key_layer1,
                "attn1": attention_probs1,
                "querues1": query_layer1,
                "keys2": key_layer2,
            }

        return context_layer1, context_layer2, attn_data


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BertBiOutput(nn.Module):
    def __init__(self, hidden_size=512,dropout_rate=0.1):
        super(BertBiOutput, self).__init__()

        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.q_dense1 = nn.Linear(hidden_size, hidden_size)
        self.q_dropout1 = nn.Dropout(dropout_rate)

        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.q_dense2 = nn.Linear(hidden_size, hidden_size)
        self.q_dropout2 = nn.Dropout(dropout_rate)

    def forward(self, hidden_states1, input_tensor1, hidden_states2, input_tensor2):

        context_state1 = self.dense1(hidden_states1)
        context_state1 = self.dropout1(context_state1)
        context_state2 = self.dense2(hidden_states2)
        context_state2 = self.dropout2(context_state2)
        hidden_states1 = self.LayerNorm1(context_state1 + input_tensor1)
        hidden_states2 = self.LayerNorm2(context_state2 + input_tensor2)
        return hidden_states1, hidden_states2

class BertImageIntermediate(nn.Module):
    def __init__(self, hidden_size):
        super(BertImageIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertImageOutput(nn.Module):
    def __init__(self, hidden_size=512,dropout_rate=0.1):
        super(BertImageOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertIntermediate(nn.Module):
    def __init__(self, hidden_size):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, hidden_size=512,dropout_rate=0.1):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Co_attention_block(nn.Module):
    def __init__(self,num_attention_heads=6,hidden_size=512,dropout_rate=0.1):
        super(Co_attention_block, self).__init__()
        self.biattention = BertBiAttention(num_attention_heads,hidden_size,dropout_rate)

        self.biOutput = BertBiOutput(hidden_size,dropout_rate)

        self.v_intermediate = BertImageIntermediate(hidden_size)
        self.v_output = BertImageOutput(hidden_size,dropout_rate)

        self.t_intermediate = BertIntermediate(hidden_size)
        self.t_output = BertOutput(hidden_size,dropout_rate)

    def forward(
        self,
        vision_input_tensor,
        text_input_tensor,
    ):

        text_bi_output, vision_bi_output, co_attention_probs = self.biattention(
            vision_input_tensor,
            text_input_tensor,)
        vision_attention_output, text_attention_output = self.biOutput(
            vision_bi_output, vision_input_tensor, text_bi_output, text_input_tensor)
        vision_intermediate_output = self.v_intermediate(vision_attention_output)
        vision_layer_output = self.v_output(vision_intermediate_output, vision_attention_output)
        text_intermediate_output = self.t_intermediate(text_attention_output)
        text_layer_output = self.t_output(text_intermediate_output, text_attention_output)
        return vision_layer_output, text_layer_output, co_attention_probs
    
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.n_head = n_head

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor):
        attn_mask_ = attn_mask.repeat_interleave(self.n_head, dim=0)
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, para_tuple: tuple):
        # x: torch.Tensor, attn_mask: torch.Tensor
        # print(para_tuple)
        x, attn_mask = para_tuple
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, attn_mask)
    
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
    
