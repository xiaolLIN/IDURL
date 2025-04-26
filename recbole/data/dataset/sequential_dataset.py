# @Time   : 2020/9/16
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

# UPDATE:
# @Time   : 2020/9/16, 2021/7/1, 2021/7/11
# @Author : Yushuo Chen, Xingyu Pan, Yupeng Hou
# @Email  : chenyushuo@ruc.edu.cn, xy_pan@foxmail.com, houyupeng@ruc.edu.cn

"""
recbole.data.sequential_dataset
###############################
"""

import numpy as np
import torch

from recbole.data.dataset import Dataset
from recbole.data.interaction import Interaction
from recbole.utils.enum_type import FeatureType, FeatureSource
from collections import Counter


class SequentialDataset(Dataset):
    """:class:`SequentialDataset` is based on :class:`~recbole.data.dataset.dataset.Dataset`,
    and provides augmentation interface to adapt to Sequential Recommendation,
    which can accelerate the data loader.

    Attributes:
        max_item_list_len (int): Max length of historical item list.
        item_list_length_field (str): Field name for item lists' length.
    """

    def __init__(self, config):
        self.max_item_list_len = config['MAX_ITEM_LIST_LENGTH']
        self.item_list_length_field = config['ITEM_LIST_LENGTH_FIELD']
        super().__init__(config)
        if config['benchmark_filename'] is not None:
            self._benchmark_presets()

    def _change_feat_format(self):
        """Change feat format from :class:`pandas.DataFrame` to :class:`Interaction`,
           then perform data augmentation.
        """
        super()._change_feat_format()

        if self.config['benchmark_filename'] is not None:
            return
        self.logger.debug('Augmentation for sequential recommendation.')
        self.data_augmentation()

    def _aug_presets(self):
        list_suffix = self.config['LIST_SUFFIX']
        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = field + list_suffix
                setattr(self, f'{field}_list_field', list_field)
                ftype = self.field2type[field]

                if ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ]:
                    list_ftype = FeatureType.TOKEN_SEQ
                else:
                    list_ftype = FeatureType.FLOAT_SEQ

                if ftype in [FeatureType.TOKEN_SEQ, FeatureType.FLOAT_SEQ]:
                    list_len = (self.max_item_list_len, self.field2seqlen[field])
                else:
                    list_len = self.max_item_list_len

                self.set_field_property(list_field, list_ftype, FeatureSource.INTERACTION, list_len)

        self.set_field_property(self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1)

    def data_augmentation(self):
        """Augmentation processing for sequential dataset.

        E.g., ``u1`` has purchase sequence ``<i1, i2, i3, i4>``,
        then after augmentation, we will generate three cases.

        ``u1, <i1> | i2``

        (Which means given user_id ``u1`` and item_seq ``<i1>``,
        we need to predict the next item ``i2``.)

        The other cases are below:

        ``u1, <i1, i2> | i3``

        ``u1, <i1, i2, i3> | i4``
        """
        self.logger.debug('data_augmentation')

        self._aug_presets()

        self._check_field('uid_field', 'time_field')
        max_item_list_len = self.config['MAX_ITEM_LIST_LENGTH']
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        uid_list = np.array(uid_list)
        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)

        new_length = len(item_list_index)
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
        }

        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f'{field}_list_field')
                list_len = self.field2seqlen[list_field]
                shape = (new_length, list_len) if isinstance(list_len, int) else (new_length,) + list_len
                list_ftype = self.field2type[list_field]
                dtype = torch.int64 if list_ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ] else torch.float64
                new_dict[list_field] = torch.zeros(shape, dtype=dtype)

                value = self.inter_feat[field]
                for i, (index, length) in enumerate(zip(item_list_index, item_list_length)):
                    new_dict[list_field][i][:length] = value[index]

        # interest drift quantization, calculate the IDM, transforms into IDQ
        # =====
        old_repr = 1
        newc_degrees = []
        target_items = self.inter_feat[target_index][self.iid_field]
        item_seqs = new_dict[getattr(self, self.iid_field + '_list_field')]
        item_cates = self.get_item_feature()['categories']
        item_seq_lens = new_dict[self.item_list_length_field]
        for i, uid in enumerate(uid_list):
            target_item = target_items[i]
            tar_item_cate = item_cates[target_item]
            seq_len = item_seq_lens[i]
            item_seq = item_seqs[i][:seq_len]

            if old_repr == 1 and target_item in list(item_seq.numpy()):
                newc_degrees.append(0)
                continue

            item_seq_cates_set = set((item_cates[item_seq]).numpy().reshape(-1))
            # 0 for removing the padding token, 1 for removing "Beauty" or "Sports", such a large category (dataset name)
            item_seq_cates_set = item_seq_cates_set - {0} if self.config['dataset'] == 'yelp' else item_seq_cates_set - {0, 1}

            tar_item_cate_set = set(tar_item_cate.numpy().reshape(-1))
            tar_item_cate_set = tar_item_cate_set - {0} if self.config['dataset'] == 'yelp' else tar_item_cate_set - {0, 1}
            tar_item_cate_num = len(tar_item_cate_set)

            same_cate_num = len(item_seq_cates_set & tar_item_cate_set)
            same_ratio = same_cate_num / tar_item_cate_num if tar_item_cate_num != 0 else 0

            if old_repr == 1:
                if self.config['n_newc_repr'] == 2:
                    if same_ratio == 0:
                        newc_degrees.append(2)
                    else:
                        newc_degrees.append(1)
                elif self.config['n_newc_repr'] == 3:
                    if same_ratio == 0:
                        newc_degrees.append(3)
                    elif same_ratio == 1:
                        newc_degrees.append(1)
                    else:
                        newc_degrees.append(2)
                elif self.config['n_newc_repr'] == 4:
                    if same_ratio == 0:
                        newc_degrees.append(4)
                    elif same_ratio < 0.5:
                        newc_degrees.append(3)
                    elif same_ratio < 1:
                        newc_degrees.append(2)
                    else:
                        newc_degrees.append(1)
                elif self.config['n_newc_repr'] == 5:
                    if same_ratio == 0:
                        newc_degrees.append(5)
                    elif same_ratio < 0.33:
                        newc_degrees.append(4)
                    elif same_ratio < 0.66:
                        newc_degrees.append(3)
                    elif same_ratio < 1:
                        newc_degrees.append(2)
                    else:
                        newc_degrees.append(1)

        new_dict['newc_degree'] = torch.tensor(newc_degrees)
        new_data.update(Interaction(new_dict))
        # =====

        # IDRA sampling
        if self.config['idra'] == 1:
            same_target_i_c_index = self.semantic_augmentation(new_data)
            null_index = []
            sample_pos = []
            for i, targets in enumerate(same_target_i_c_index):
                if len(targets) == 0:
                    sample_pos.append(-1)
                    null_index.append(i)
                else:
                    sample_pos.append(np.random.choice(targets))

            sem_pos_seqs = new_data[getattr(self, self.iid_field + '_list_field')][sample_pos]
            sem_pos_lengths = new_data['item_length'][sample_pos]

            sem_aug_user_ids = new_data['user_id'][sample_pos]

            if null_index:
                if self.config['dataset'] == 'yelp':
                    sem_pos_seqs[null_index] = new_data['business_id_list'][null_index]
                else:
                    sem_pos_seqs[null_index] = new_data['item_id_list'][null_index]

                sem_pos_lengths[null_index] = new_data['item_length'][null_index]

                sem_aug_user_ids[null_index] = new_data['user_id'][null_index]

            new_data.update(Interaction({'sem_aug': sem_pos_seqs, 'sem_aug_lengths': sem_pos_lengths,
                                         'sem_aug_user_ids': sem_aug_user_ids}))


        self.inter_feat = new_data

    def semantic_augmentation(self, aug_seqs):
        aug_path = self.config['data_path'] + '/semantic_augmentation.npy'
        import os
        if os.path.exists(aug_path):
            same_target_index = np.load(aug_path, allow_pickle=True)
        else:
            same_target_index = []
            if self.config['dataset'] == 'yelp':
                target_item = aug_seqs['business_id'].numpy()
            else:
                target_item = aug_seqs['item_id'].numpy()

            target_newc_degree = aug_seqs['newc_degree'].numpy()

            for index, (item_id, newc_degree) in enumerate(zip(target_item, target_newc_degree)):
                all_index_same_id = np.where(target_item == item_id)[0]  # all index of a specific item id with self item
                delete_index = np.argwhere(all_index_same_id == index)
                all_index_same_id_wo_self = np.delete(all_index_same_id, delete_index)

                all_index_same_newcdegree = np.where(target_newc_degree == newc_degree)[0]
                delete_index = np.argwhere(all_index_same_newcdegree == index)
                all_index_same_newcdegree_wo_self = np.delete(all_index_same_newcdegree, delete_index)

                all_index_same_id_wo_self = np.array(list(set(all_index_same_id_wo_self) & set(all_index_same_newcdegree_wo_self)))
                same_target_index.append(all_index_same_id_wo_self)

            same_target_index = np.array(same_target_index)
            np.save(aug_path, same_target_index)

        return same_target_index

    def _benchmark_presets(self):
        list_suffix = self.config['LIST_SUFFIX']
        for field in self.inter_feat:
            if field + list_suffix in self.inter_feat:
                list_field = field + list_suffix
                setattr(self, f'{field}_list_field', list_field)
        self.set_field_property(self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1)
        self.inter_feat[self.item_list_length_field] = self.inter_feat[self.item_id_list_field].agg(len)

    def inter_matrix(self, form='coo', value_field=None):
        """Get sparse matrix that describe interactions between user_id and item_id.
        Sparse matrix has shape (user_num, item_num).
        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = self.inter_feat[src, tgt]``.

        Args:
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        if not self.uid_field or not self.iid_field:
            raise ValueError('dataset does not exist uid/iid, thus can not converted to sparse matrix.')

        l1_idx = (self.inter_feat[self.item_list_length_field] == 1)
        l1_inter_dict = self.inter_feat[l1_idx].interaction
        new_dict = {}
        list_suffix = self.config['LIST_SUFFIX']
        candidate_field_set = set()
        for field in l1_inter_dict:
            if field != self.uid_field and field + list_suffix in l1_inter_dict:
                candidate_field_set.add(field)
                new_dict[field] = torch.cat([self.inter_feat[field], l1_inter_dict[field + list_suffix][:, 0]])
            elif (not field.endswith(list_suffix)) and (field != self.item_list_length_field):
                new_dict[field] = torch.cat([self.inter_feat[field], l1_inter_dict[field]])
        local_inter_feat = Interaction(new_dict)
        return self._create_sparse_matrix(local_inter_feat, self.uid_field, self.iid_field, form, value_field)

    def build(self):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~recbole.config.eval_setting.EvalSetting` for details.

        Args:
            eval_setting (:class:`~recbole.config.eval_setting.EvalSetting`):
                Object contains evaluation settings, which guide the data processing procedure.

        Returns:
            list: List of built :class:`Dataset`.
        """
        ordering_args = self.config['eval_args']['order']
        if ordering_args != 'TO':
            raise ValueError(f'The ordering args for sequential recommendation has to be \'TO\'')

        return super().build()
