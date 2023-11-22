import numpy as np
import pandas as pd
import os

class BertInputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        
    def convert_examples_to_features(df, max_length, tokenizer):
        # df= train_dataset.dataset
        
        features = []
        for i, row in df.iterrows():
            first_tokens = tokenizer.tokenize(row['Question'])
            sec_tokens = tokenizer.tokenize(row['Answer'])
            tokens = ["[CLS]"] + first_tokens + ["[SEP]"] + sec_tokens
            if len(sec_tokens) + len(first_tokens) > max_length -1:
                tokens = tokens[:(max_length-1)]
            tokens = tokens + ['[SEP]']
    
            segment_ids = [0] * (len(first_tokens)+2)
            segment_ids += [1] * (len(sec_tokens)+1)
    
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
            input_mask = [1] * len(input_ids)
    
            padding = [0] * (max_length - len(input_ids))
            input_ids += padding 
            input_mask += padding
            segment_ids += padding
    
            assert len(segment_ids) == max_length
            assert len(input_ids) == max_length
            assert len(input_mask) == max_length
            
            features.append(
                BertInputFeatures(
                    input_ids = input_ids,
                    input_mask = input_mask,
                    segment_ids = segment_ids
                    
                )
            )
        return features
            
