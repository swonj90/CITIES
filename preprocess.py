from tqdm import tqdm
tqdm.pandas()
import pandas as pd
import numpy as np
from config import Config
from utils import Utils, ItemDictionary

class Preprocess:

    def __init__(self, args, config: Config, utils: Utils):
        self.args = args
        self.utils = utils
        self.config = config
        self.max_len = config.MAX_LEN
        self.pad = config.PAD
        self.freq_lbound = config.FREQ_LBOUND
        self.val_ratio = config.VAL_RATIO

    def make_inference_data(self, train_context, item_dict, mask):
        remove_items = {}
        for _, item in item_dict.items():
            if self.item_dictionary.item_count[item] <= self.config.FREQ_LBOUND:
                '''
                    Only Choose items with sufficient occurance (so that we can guarantee the groundtruth embedding)
                '''
                remove_items[item] = True

        inference_dataset = {item: [[], []] for item in remove_items}
        inference_dataset = self.make_context(train_context, inference_dataset, mask)

        return inference_dataset

    def make_context(self, context_dict, context_dataset, mask):
        for user in context_dict:
            items = []
            seq = context_dict[user]
            for idx, item in enumerate(seq):
                if item in context_dataset:
                    items += [[item, idx]]
            if len(items) > 0:
                seq_ids = seq
                for item, idx in items:
                    if len(seq_ids[max(0, idx-self.max_len): idx+1+self.max_len]) > 2:
                        context_dataset[item][0] += [
                            seq_ids[max(0, idx-self.max_len): idx]]
                        context_dataset[item][1] += [
                            seq_ids[idx+1:  idx+1+self.max_len]]

        context_dataset = self.mask_to_target(mask, context_dataset)
        return context_dataset

    def filtering_extreme_cold_user_item(self, df):
        item_sizes = df.groupby('sid').size()
        good_items = item_sizes.index[item_sizes >= 2]
        df = df[df['sid'].isin(good_items)]
        user_sizes = df.groupby('uid').size()
        good_users = user_sizes.index[user_sizes >= 5]
        df = df[df['uid'].isin(good_users)]
        return df

    def load_data(self):
        df = pd.read_csv('data/{}.csv'.format(self.args.dataset))
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        df = df[df['rating'] >= 4]
        
        df = self.filtering_extreme_cold_user_item(df)
        umap = {u: i+1 for i, u in enumerate(set(df['uid']))}
        smap = {s: i+1 for i, s in enumerate(set(df['sid']))}
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        
        train, test = self.split_df(df, umap)
        return train, smap
    
    def split_df(self, df, umap):
        user_group = df.groupby('uid')
        user2items = user_group.progress_apply(lambda d: list(d.sort_values(by='timestamp')['sid']))
        train, test = {}, {}
        user_count = len(umap)
        split_point = int(user_count*0.8)
        for user in range(1, user_count+1):
            items = user2items[user]
            if user < split_point:
                train[user] = items
            else:
                test[user] = items
        return train, test   

    def make_training_data(self, train_context, item_dict, item_embedding, mask):
        self.item_dictionary = ItemDictionary(item_embedding)
        for u in train_context:
            for i in train_context[u]:
                self.item_dictionary.item_count[i] += 1

        remove_items = {}
        for _, item in item_dict.items():
            if self.item_dictionary.item_count[item] > self.freq_lbound:
                '''
                    Only Choose items with sufficient occurance (so that we can guarantee the groundtruth embedding)
                '''
                remove_items[item] = True

        train_dataset, valid_dataset = {}, {}
        for item, prob in zip(remove_items, np.random.random(len(remove_items))):
            if prob >= self.val_ratio:
                '''
                    Use 95% for training and 5% for validation
                '''
                train_dataset[item] = [[], []]
            else:
                valid_dataset[item] = [[], []]

        train_dataset = self.make_context(train_context, train_dataset, mask)
        valid_dataset = self.make_context(train_context, valid_dataset, mask)

        return train_dataset, valid_dataset

    def mask_to_target(self, mask, context_dataset):
        items = list(context_dataset.keys())
        for item in items:
            if context_dataset[item][0]:
                lefts = self.utils.pad_sequences(
                    context_dataset[item][0], maxlen=self.max_len, value=self.pad, padding='pre', truncating='pre')
                rights = self.utils.pad_sequences(
                    context_dataset[item][1], maxlen=self.max_len, value=self.pad, padding='post', truncating='post')
                center = np.array([[mask] for i in range(len(rights))])
                context_dataset[item] = np.concatenate(
                    (lefts, center, rights), axis=1)
            else:
                del context_dataset[item]
        return context_dataset
