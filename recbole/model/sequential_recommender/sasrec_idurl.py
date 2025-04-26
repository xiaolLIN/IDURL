import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss


class SASRec_IDURL(SequentialRecommender):
    def __init__(self, config, dataset):
        super(SASRec_IDURL, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.device = config['device']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=0)
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.K = config['n_facet_all']       # the number of interest drift levels considered by IDURL
        self.project_arr = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=True) for i in range(self.K)])
        self.after_proj_ln_arr = nn.ModuleList([nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps) for i in range(self.K)])

        self.repr_dp = config['new_repr_dp']
        if self.repr_dp:
            self.repr_dropout = nn.Dropout(self.hidden_dropout_prob)

        self.new_repr_ln_arr = nn.ModuleList(nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps) for i in range(self.K))
        self.idra_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.disen_loss_fct = nn.CrossEntropyLoss()
        self.disen_lambda = config['disen_lambda']

        self.idra = config['idra']
        self.batch_size = config['train_batch_size']
        if self.idra == 1:
            self.align_lambda = config['align_lambda']
            self.tau = 1
            self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
            self.cl_loss_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)
        # self.other_parameter_name = ['feature_embed_layer_list']

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

    def get_proj_repr(self, input_emb, i):
        return self.project_arr[i](input_emb)

    def get_layer_norm(self, input_emb, i):
        return self.after_proj_ln_arr[i](input_emb)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def cl_task(self, z_i, z_j, temp, batch_size, sim_computer='dot'):
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)  # [2B, d]

        if sim_computer == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim_computer == 'dot':
            sim = torch.mm(z, z.T) / temp            # [2B, 2B]

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def get_repr_list(self, ori_u_repr):
        repr_list = []
        for i in range(self.K):
            projected_repr = self.get_proj_repr(ori_u_repr, i)
            projected_repr_dp = self.repr_dropout(projected_repr)
            drift_repr = self.get_layer_norm(projected_repr_dp + ori_u_repr, i)
            repr_list.append(drift_repr)

        drift_repr_list = []
        for i, new_repr in enumerate(repr_list):
            repr = self.new_repr_ln_arr[i](self.repr_dropout(new_repr))
            drift_repr_list.append(repr.unsqueeze(-1))  # lists   K * [B,1,D,1]

        return drift_repr_list


    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        return trm_output  # [B L H]

    def calculate_loss_prob(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        target_item_drift_degree = interaction['newc_degree']  # [B]

        # User Representation Encoding  with SASRec as the backbone model
        all_hidden_states = self.forward(item_seq, item_seq_len)
        seq_output = all_hidden_states[-1]
        ori_u_repr = self.gather_indexes(seq_output, item_seq_len - 1).unsqueeze(dim=1)  # [B,L,D] gather index -> [B,D] -> [B,1,D]

        # Drift Representation Generation (DRG)
        drift_repr_list = self.get_repr_list(ori_u_repr)  # lists   K * [B,1,D,1]

        loss = 0

        # Interest Drift-aware Representation Alignment (IDRA)
        if self.idra == 1 and self.training:
            # unsupervised augmentation (ua) view
            repr_sum = torch.sum(torch.cat(drift_repr_list, dim=1), dim=1).squeeze(-1)  # [B,n,D,1] -> [B,D,1] -> [B,D]
            un_aug_repr = self.idra_ln(self.dropout(repr_sum))

            # supervised sampling (ss) view
            su_aug_seq, su_aug_lengths = interaction['sem_aug'], interaction['sem_aug_lengths']
            su_hidden_states = self.forward(su_aug_seq, su_aug_lengths)
            su_aug_seq_output = su_hidden_states[-1]
            su_aug_u_repr = self.gather_indexes(su_aug_seq_output, su_aug_lengths - 1).unsqueeze(dim=1)  # [B,L,D] -> [B,D] -> [B,1,D]
            su_aug_drift_repr_list = self.get_repr_list(su_aug_u_repr)  # lists   K * [B,1,D,1]
            su_aug_repr_sum = torch.sum(torch.cat(su_aug_drift_repr_list, dim=1), dim=1).squeeze(-1)  # [B,n,D,1] -> [B,D,1] -> [B,D]
            su_aug_repr = self.idra_ln(su_aug_repr_sum)

            # cl task & cl loss
            cl_logits, cl_labels = self.cl_task(un_aug_repr, su_aug_repr, temp=self.tau, batch_size=item_seq_len.shape[0])
            loss += self.align_lambda * self.cl_loss_fct(cl_logits, cl_labels)


        # Interest Drift-guided Representation Disentanglement (IDRD)
        if self.disen_lambda > 0 and self.training:
            labels = target_item_drift_degree  # [B]
            target_item_emb = self.item_embedding(pos_items).unsqueeze(1)  # [B,D]->[B,1,D]
            drift_reprs = torch.cat(drift_repr_list, dim=1).squeeze(-1)  # [B, K, D]
            diseng_logits = (target_item_emb * drift_reprs).sum(-1)  # [B, K]
            disen_loss = self.disen_loss_fct(diseng_logits.view(-1, self.K), labels)
            loss += self.disen_lambda * disen_loss

        # Prediction   this parallelizes all the candidate item.
        logits_list = []
        for drift_repr in drift_repr_list:
            logits = F.linear(drift_repr.squeeze(-1), self.item_embedding.weight, None)  # [B,1,D,1] -> [B,1,D]-> [B,1,|V| ]
            logits_list.append(logits.unsqueeze(dim=-1))  # lists   K * [B,1,|V|,1]

        # item embedding as query
        candi_query_logits = torch.cat(logits_list, dim=-1)  # [B,1,|V|, K]
        # interest drift distribution about the candidate items
        candi_id_distr = candi_query_logits.softmax(dim=-1)   # [B,1,|V|, K]
        final_logits = (candi_id_distr * candi_query_logits).sum(dim=-1)  # [B, 1, |V| ]

        prediction_prob = final_logits.softmax(dim=-1)

        # Recommendation Task & loss
        if item_seq is not None:
            inp = torch.log(prediction_prob.view(-1, self.n_items) + 1e-8)
            loss_raw = self.loss_fct(inp, pos_items.view(-1))
            loss += loss_raw.mean()
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
