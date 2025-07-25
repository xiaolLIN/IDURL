import torch
from torch import nn
import math
import copy
from recbole.model.layers import FeatureSeqEmbLayer, FeedForward, VanillaAttention
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, hidden_size, attribute_hidden_size, feat_num, hidden_dropout_prob, attn_dropout_prob,
                 layer_norm_eps, fusion_type, max_len):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.attribute_attention_head_size = [int(_ / n_heads) for _ in attribute_hidden_size]
        self.attribute_all_head_size = [self.num_attention_heads * _ for _ in self.attribute_attention_head_size]

        self.fusion_type = fusion_type
        self.max_len = max_len
        self.feat_num = feat_num

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        if self.fusion_type == 'concat':
            self.fusion_layer = nn.Linear(hidden_size * (2+self.feat_num), hidden_size)
        elif self.fusion_type == 'gate':
            self.fusion_layer = VanillaAttention(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attribute_table, position_embedding, attention_mask):
        item_value_layer = self.value(input_tensor)

        # input_tensor  [B, L, d_f]
        # attribute_table [B, L, 1, d_f]  的list
        # position_embedding  [B, L, d_f]

        attribute_table = torch.cat(attribute_table, dim=-2)    # [B, L, 1, d_f] 的 list， 变成 [B, L, fea num, d_f] 的tensor
        table_shape = attribute_table.shape
        feat_num, attr_hidden_size = table_shape[-2], table_shape[-1]

        if self.fusion_type == 'sum':
            mixed_side_emb = torch.cat((attribute_table, input_tensor.unsqueeze(-2),
                                        position_embedding.unsqueeze(-2)), dim=-2)  # [B, L, fea num +2, d]
            mixed_side_emb = torch.sum(mixed_side_emb, dim=-2)   # [B, L, d]
        elif self.fusion_type == 'concat':
            attr_emb = attribute_table.view(table_shape[:-2] + (feat_num * attr_hidden_size, ))   # [B,L,fea num * d_f]
            mixed_side_emb = torch.cat((attr_emb, input_tensor, position_embedding), dim=-1)
            mixed_side_emb = self.fusion_layer(mixed_side_emb)           # [B, L, D]
        elif self.fusion_type == 'gate':
            mixed_side_emb = torch.cat((attribute_table, input_tensor.unsqueeze(-2),
                                        position_embedding.unsqueeze(-2)), dim=-2)    # [B, L, fea num +2, d]
            mixed_side_emb, _ = self.fusion_layer(mixed_side_emb)

        mixed_query_layer = self.query(mixed_side_emb)
        mixed_key_layer = self.key(mixed_side_emb)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(item_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TransformerLayer(nn.Module):

    def __init__(
            self, n_heads, hidden_size, attribute_hidden_size, feat_num, intermediate_size, hidden_dropout_prob,
            attn_dropout_prob, hidden_act,
            layer_norm_eps, fusion_type, max_len
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, attribute_hidden_size, feat_num, hidden_dropout_prob, attn_dropout_prob,
            layer_norm_eps, fusion_type, max_len,
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, attribute_embed, position_embedding, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attribute_embed, position_embedding, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        attribute_hidden_size=[64],
        feat_num=1,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12,
        fusion_type='sum',
        max_len=None
    ):

        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(
            n_heads, hidden_size, attribute_hidden_size, feat_num, inner_size, hidden_dropout_prob, attn_dropout_prob,
            hidden_act, layer_norm_eps, fusion_type, max_len
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attribute_hidden_states, position_embedding, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attribute_hidden_states, position_embedding, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class NOVA(SequentialRecommender):

    def __init__(self, config, dataset):
        super(NOVA, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.attribute_hidden_size = [config['hidden_size']]
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.selected_features = config['selected_features']
        self.pooling_mode = config['pooling_mode']
        self.device = config['device']
        self.num_feature_field = len(config['selected_features'])

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        self.fusion_type = config['fusion_type']

        # self.lamdas = config['lamdas']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        # self.feature_embed_layer_list = nn.ModuleList(
        #     [copy.deepcopy(FeatureSeqEmbLayer(dataset, self.attribute_hidden_size[_], [self.selected_features[_]],
        #                                       self.pooling_mode, self.device)) for _
        #      in range(len(self.selected_features))])
        self.feature_embed_layer_list = nn.ModuleList(
            [FeatureSeqEmbLayer(dataset,self.attribute_hidden_size[0],[self.selected_features[0]],self.pooling_mode,self.device)])

        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            attribute_hidden_size=self.attribute_hidden_size,
            feat_num=len(self.selected_features),
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            fusion_type=self.fusion_type,
            max_len=self.max_seq_length
        )

        self.n_attributes = {}
        for attribute in self.selected_features:
            self.n_attributes[attribute] = len(dataset.field2token_id[attribute])

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            # self.loss_fct = nn.CrossEntropyLoss()
            self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=0)  # modified for mfs
            # self.attribute_loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # ======================
        self.n_facet_all = config['n_facet_all']  # added for mfs
        self.n_facet_effective = 1
        self.n_embd = self.hidden_size
        self.use_proj_bias = config['use_proj_bias']
        self.project_arr = nn.ModuleList([nn.Linear(self.n_embd, self.n_embd, bias=self.use_proj_bias) for i in range(self.n_facet_all)])
        self.after_proj_ln_arr = nn.ModuleList([nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps) for i in range(self.n_facet_all)])

        self.new_repr_dp = config['new_repr_dp']
        if self.new_repr_dp:
            self.new_repr_dropout = nn.Dropout(self.hidden_dropout_prob)
        self.old_repr_dp = config['old_repr_dp']
        if self.old_repr_dp:
            self.old_repr_dropout = nn.Dropout(self.hidden_dropout_prob)

        # ======================

        # parameters initialization
        self.apply(self._init_weights)
        self.other_parameter_name = ['feature_embed_layer_list']

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_facet_emb(self, input_emb, i):
        return self.project_arr[i](input_emb)

    def get_layer_norm(self, input_emb, i):
        return self.after_proj_ln_arr[i](input_emb)

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        feature_table = []
        for feature_embed_layer in self.feature_embed_layer_list:
            sparse_embedding, dense_embedding = feature_embed_layer(None, item_seq)
            sparse_embedding = sparse_embedding['item']
            dense_embedding = dense_embedding['item']
            # concat the sparse embedding and float embedding
            if sparse_embedding is not None:
                feature_table.append(sparse_embedding)
            if dense_embedding is not None:
                feature_table.append(dense_embedding)

        feature_emb = feature_table
        input_emb = item_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, feature_emb, position_embedding, extended_attention_mask, output_all_encoded_layers=True)
        # output = trm_output[-1]
        # output = self.gather_indexes(output, item_seq_len - 1)
        # return output  # [B H]
        return trm_output  # [B L H]

    def calculate_loss_prob(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        all_hidden_states = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        seq_output = all_hidden_states[-1]
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1).unsqueeze(dim=1)  # [B,L,D] gether index -> [B,D] ->[B,1,D]

        projected_emb = self.get_facet_emb(seq_output, 0)  # (bsz, 1, hidden_dim)
        projected_emb_dp = self.new_repr_dropout(projected_emb)
        seq_output = self.get_layer_norm(projected_emb_dp + seq_output, 0)

        logits = F.linear(seq_output, self.item_embedding.weight, None)
        logits_softmax = logits.softmax(dim=-1)
        prediction_prob = logits_softmax
        if item_seq is not None:
            inp = torch.log(prediction_prob.view(-1, self.n_items) + 1e-8)
            loss_raw = self.loss_fct(inp, pos_items.view(-1))
            loss = loss_raw.mean()
        else:
            raise Exception("Labels can not be None")
        return loss, prediction_prob.squeeze(dim=1)

    def calculate_loss(self, interaction):
        loss, _ = self.calculate_loss_prob(interaction)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        _, prediction_prob = self.calculate_loss_prob(interaction)
        return prediction_prob
