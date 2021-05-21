from collections import defaultdict
import numpy as np
import torch
from torch.autograd import Variable

class LambdaLR:
    def __init__(self, n_steps, decay_start_step):
        self.n_steps = n_steps
        self.decay_start_step = decay_start_step

    def step(self, step):
        return 1.0 - max(0, step - self.decay_start_step) / (self.n_steps - self.decay_start_step)

class ItemDictionary:
    def __init__(self, item_embedding):
        self.item_count = defaultdict(int)
        self.len = item_embedding.shape[0]

    def __len__(self):
        return self.len

class Utils:
    def __init__(self, config):
        self.config = config

    def bucketting(self, items, dataset):
        n_contexts = dict()
        for item in items:
            if len(dataset[item]) != 0:
                sample_sents = dataset[item]
                n_ctx = len(sample_sents)
                if n_ctx in n_contexts:
                    tmp = n_contexts[n_ctx]
                    tmp.append(item)
                    n_contexts[n_ctx] = tmp
                else:
                    n_contexts[n_ctx] = [item]
        return n_contexts


    def get_item_embedding(self, args):
        pretrained_model = torch.load('pretrained_model/{}/{}.pth'.format(args.dataset, args.pretrained_model))
        item_embedding = pretrained_model.get('model_state_dict')[args.embedding_key].cpu().numpy()
        return item_embedding, pretrained_model

    def pad_sequences(self, sequences, maxlen=None, dtype='int32',
                      padding='pre', truncating='pre', value=0):
        '''
            Directly adopted from keras.preprocessing
        '''
        num_samples = len(sequences)
        lengths = []
        for x in sequences:
            lengths.append(len(x))

        sample_shape = tuple()
        for s in sequences:
            if len(s) > 0:
                sample_shape = np.asarray(s).shape[1:]
                break
        x = np.full((num_samples, maxlen) +
                    sample_shape, value, dtype=dtype)
        for idx, s in enumerate(sequences):
            if not len(s):
                continue  # empty list/array was found
            if truncating == 'pre':
                trunc = s[-maxlen:]
            elif truncating == 'post':
                trunc = s[:maxlen]
            else:
                raise ValueError('Truncating type "%s" '
                                    'not understood' % truncating)

            # check `trunc` has expected shape
            trunc = np.asarray(trunc, dtype=dtype)

            if padding == 'post':
                x[idx, :len(trunc)] = trunc
            elif padding == 'pre':
                x[idx, -len(trunc):] = trunc
        return x


    def get_train_batch(self, items, dataset, item2vec, batch_size, k_shot):
        items = np.random.choice(items, batch_size)
        contexts, targets = [], []
        for item in items:
            if len(dataset[item]) != 0:
                sample_seq_idx = np.random.choice(
                    len(dataset[item]), k_shot)
                seqs = dataset[item][sample_seq_idx]
                contexts += [seqs]
                targets += [item2vec[item]]
        contexts = torch.LongTensor(contexts).cuda()
        targets = Variable(torch.FloatTensor(targets).cuda())
        return contexts, targets

    def get_inference_batch(self, items, dataset):
        contexts, targets = [], []
        for item in items:
            if len(dataset[item]) != 0:
                seqs = dataset[item]
                contexts += [seqs]
                targets += [item]
        contexts = torch.LongTensor(contexts).cuda()
        targets = torch.LongTensor(targets).cuda()
        return contexts, targets
