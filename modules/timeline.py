import torch
import numpy as np
import torch.nn as nn


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        # self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        # x = self.layer_norm(x)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn)  # [n * b, l_q, l_k]
        attn = self.dropout(attn)  # [n * b, l_v, d]

        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)

        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        # output = self.layer_norm(output)

        return output, attn


class TimeAggModel(torch.nn.Module):
    def __init__(self, time_line_length, feat_dim, agg_time_line='linear', use_global=False, use_extra_time=False):
        super(TimeAggModel, self).__init__()
        self.agg_time_line = agg_time_line
        self.time_line_length = time_line_length
        self.feat_dim = feat_dim
        self.use_global = use_global
        self.use_extra_time = use_extra_time
        if self.agg_time_line == 'linear':
            self.aggregate_timelines = \
                torch.nn.Linear(
                    (1 + (1 + self.use_global + self.use_extra_time) * self.time_line_length) * self.feat_dim,
                    self.feat_dim
                )
        elif self.agg_time_line == 'self-regression':
            self.aggregate_timelines = torch.nn.LSTM(self.feat_dim, self.feat_dim, num_layers=1, batch_first=True)
            # self.aggregate_timelines = None
        elif self.agg_time_line == 'auto-encoder':
            n_head = 1
            self.aggregate_timelines = MultiHeadAttention(n_head=n_head, d_model=self.feat_dim,
                                                          d_k=self.feat_dim // n_head, d_v=self.feat_dim // n_head,
                                                          dropout=0.1)
            self.merger = MergeLayer(dim1=self.feat_dim, dim2=self.feat_dim, dim3=self.feat_dim, dim4=self.feat_dim)
        else:
            exit('ERROR, aggregate timeline method not specified')
            self.aggregate_timelines = None

    def forward(self, cur_local_embed, prev_local_embeds):
        if self.agg_time_line == 'linear':
            time_line_input = torch.cat([cur_local_embed] + prev_local_embeds, dim=1)
            return self.aggregate_timelines(time_line_input)
        elif self.agg_time_line == 'self-regression':
            cur_local_embed = cur_local_embed.unsqueeze(1)
            prev_local_embeds = [_.unsqueeze(1) for _ in prev_local_embeds]
            outputs, (h, c) = self.aggregate_timelines(torch.cat([cur_local_embed] + prev_local_embeds, dim=1))
            return outputs[:, -1]
        elif self.agg_time_line == 'auto-encoder':
            cur_local_embed = cur_local_embed.unsqueeze(1)
            prev_local_embeds = [_.unsqueeze(1) for _ in prev_local_embeds]
            inputs = torch.cat([cur_local_embed] + prev_local_embeds, dim=1)
            outputs, attn = self.aggregate_timelines(q=inputs, k=inputs, v=inputs)
            return outputs[:, 0, :]
