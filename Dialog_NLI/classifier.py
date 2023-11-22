import json
import pandas as pd
import torch
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam, BertConfig
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import f1_score
import csv
import random
import os
import matplotlib.pyplot as plt

# config: https://github.com/soroushjavdan/NLIbert/blob/master/utils/config.py

# 질문과 답변 사이의 consistency 측정



class Classifer:
    def __init__(self, bert_model = config.bert_model,gpu=False,seed=0):
        self.gpu = gpu
        sef.bert_model = bert_model
        
        self.train_df = tarin_dataset
        self.val_df = validation_dataset
        self.test_df = test_dataset
        
        self.num_classes = len(LABELS)
        
        self.model = None
        self.optimizer = None
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model)
        
        # to plot loss during training process
        self.plt_x = []
        self.plt_y = []
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.gpu:
            torch.cuda.manual_seed_all(seed)
            
    def __init_model(self):
        if self.gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        print(torch.cuda.memory_allocated(self.device))
    
    def new_model(self):
        self.model = BertForSequenceClassification.from_pretrained(
        self.bert_model, num_labels = self.num_classes
        )
        self.__init_model()
        
    def load_model(self, path_model, path_config):
        self.model = BertForSequenceClassification(BertConfig(path_config), num_labels=self.num_classes)
        self.model.load_state_dict(torch.load(path_model))
        self.__init_model()
        
    def save_model(self, path_model, path_config, epoch_n, acc, f1):
        
        # 모델 저장 경로 
        if not os.path.exists(path_model):
            os.makedirs(path_model)
            
        model_save_path= os.path.join('model_{:.4f}_{:.4f}_{:.4f}'.format(epoch_n, acc, f1))
        torch.save(self.model.state_dict(), model_save_path)
        
        # model config 파일 저장 경로
        if not os.path.exists(path_config):
            os.makedirs(path_config)
            
        model_config_path = os.path.join(path_config, 'config.cf')
        with open(model_config_path, 'w') as f: # cofig 파일 작성
            f.write(self.model.config.to_json_string())    
