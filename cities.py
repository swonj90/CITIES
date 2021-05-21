from tqdm import tqdm
import os
import numpy as np
import pytorch_warmup as warmup
import torch
import torch.nn.utils as torch_utils

from config import Config
from model import Model
from preprocess import Preprocess
from utils import LambdaLR, Utils


class CITIES:
    def __init__(self, args):
        self.args = args
        self.config = Config()
        self.utils = Utils(self.config)
        self.preprocess_class = Preprocess(self.args, self.config, self.utils)
        
    def load_data(self):
        train_context, item_dict = self.preprocess_class.load_data()
        return train_context, item_dict
        
    def preprocess(self, train_context, item_dict, is_training=True):
        """
        Preprocess
        """
        mask = len(item_dict) + 1
        if is_training:
            item_embedding, pretrained_model = self.utils.get_item_embedding(self.args)
            train_dataset, valid_dataset = self.preprocess_class.make_training_data(
                train_context, item_dict, item_embedding, mask)
            return train_dataset, valid_dataset, item_embedding, pretrained_model
        else:
            inference_dataset = self.preprocess_class.make_inference_data(train_context, item_dict, mask)
            return inference_dataset

    def train(self, train_dataset, valid_dataset, item2vec):
        """
        After applying preprocess, fit the model
        """
        source_train_items = list(train_dataset.keys())
        source_valid_items = list(valid_dataset.keys())

        best_valid_mse = -9999999999
        patience = self.config.PATIENCE
        for epoch in np.arange(self.config.N_EPOCHS):
            train_mse, valid_mse = 0, 0
            self.model.train()
            with tqdm(np.arange(int(len(train_dataset) / self.config.TRAIN_BATCH_SIZE)), desc='Train') as monitor:
                for i in monitor:
                    # randomly sample a context length, and only give the model with this size of contexts
                    k_shot = np.random.randint(
                        self.config.MAX_N_SHOT) + 1
                    train_contexts, train_targets = self.utils.get_train_batch(items=source_train_items, dataset=train_dataset,
                                                                                        item2vec=item2vec, batch_size=self.config.TRAIN_BATCH_SIZE, k_shot=k_shot)
                    self.optimizer.zero_grad()
                    pred_emb = self.model.forward(
                        train_contexts, pad=self.config.PAD)
                    loss = torch.nn.MSELoss()(pred_emb, train_targets)
                    loss.backward()
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.warmup_scheduler.dampen()
                    train_mse += -loss.cpu().detach().numpy()
                    monitor.set_postfix(train_status=train_mse)
            avg_train_mse = np.mean(train_mse)

            self.model.eval()
            with torch.no_grad():
                with tqdm(np.arange(int(len(valid_dataset) / self.config.VAL_BATCH_SIZE)), desc='Valid') as monitor:
                    for i in monitor:
                        for k_shot in np.arange(self.config.MAX_N_SHOT) + 1:
                            valid_contexts, valid_targets = self.utils.get_train_batch(items=source_valid_items, dataset=valid_dataset,
                                                                                                item2vec=item2vec, batch_size=self.config.VAL_BATCH_SIZE, k_shot=k_shot)
                            pred_emb = self.model.forward(
                                valid_contexts, pad=self.config.PAD)

                            loss = torch.nn.MSELoss()(pred_emb, valid_targets)
                            valid_mse += -loss.cpu().numpy()
                            monitor.set_postfix(valid_status=valid_mse)

            avg_valid_mse = np.mean(valid_mse)
            print(("Epoch: %d: Train MSE: %.4f; Valid MSE: %.4f; LR: %f")
                        % (epoch, avg_train_mse, avg_valid_mse, self.optimizer.param_groups[0]['lr']))

            if avg_valid_mse > best_valid_mse:
                best_valid_mse = avg_valid_mse
                inc = 0
            else:
                inc += 1
            
            if inc >= patience:
                break

    def predict(self, inference_dataset, item_embedding):
        inferencee_items = list(inference_dataset.keys())
        bucketted_items = self.utils.bucketting(
            inferencee_items, inference_dataset)

        self.model.eval()
        res_emb, res_tgt = None, None
        with torch.no_grad():
            with tqdm(bucketted_items, desc='inferencee') as monitor:
                for batch in monitor:
                    inferencee_contexts, inferencee_targets = self.utils.get_inference_batch(
                        items=bucketted_items[batch], dataset=inference_dataset)

                    num_step = int(len(inferencee_targets) /
                                    self.config.VAL_BATCH_SIZE) + 1
                    for step in range(num_step):
                        contexts = inferencee_contexts[(
                            step*self.config.VAL_BATCH_SIZE):((step+1)*self.config.VAL_BATCH_SIZE)]
                        targets = inferencee_targets[(
                            step*self.config.VAL_BATCH_SIZE):((step+1)*self.config.VAL_BATCH_SIZE)]

                        if contexts.shape[0] > 0:
                            pred_emb = self.model.forward(contexts, pad=0)
                            if res_emb is None:
                                res_emb = pred_emb
                                res_tgt = targets
                            else:
                                res_emb = torch.cat((res_emb, pred_emb))
                                res_tgt = torch.cat((res_tgt, targets))

            for w, e in zip(res_tgt.cpu().detach().numpy(), res_emb.cpu().detach().numpy()):
                item_embedding[w] = e

        return item_embedding

    def load_model(self, item_embedding):
        self.model = Model(n_head=self.config.N_HEAD, n_hid=item_embedding.shape[1], n_seq=self.config.MAX_N_SEQ,
                                    n_layer=self.config.N_LAYER, item2vec=item_embedding).cuda()
        torch_utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer = torch.optim.Adam(self.model.parameters(
        ), lr=self.config.LR, eps=self.config.EPS)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=LambdaLR(
            self.config.MAX_DECAY_STEP, self.config.DECAY_STEP).step) # MAX_DECAY_STEP > DECAY_STEP
        self.warmup_scheduler = warmup.LinearWarmup(
            self.optimizer, warmup_period=self.config.WARMUP_PERIOD)

    def update_model(self, pretrained_model, inferred_embedding):
        """
        Update the model
        """
        inferred_embedding = torch.tensor(inferred_embedding, device='cuda')
        pretrained_model['model_state_dict'].update({self.args.embedding_key: inferred_embedding})
        
        if not os.path.exists('final_model/{}'.format(self.args.dataset)):
            os.makedirs('final_model/{}'.format(self.args.dataset), exist_ok=True)
        with open('final_model/{}/final_model.pt'.format(self.args.dataset), 'wb') as f:
            torch.save(pretrained_model, f)
    
