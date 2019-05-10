import pickle
import numpy as np
from torchvision import transforms
from entity_link_data import Mention
from typing import List

# class Mention(object):
#     def __init__(self, mention:str, mention_context:str, mention_position:List[int],
#                         entity_cands:List[str], entity_context:List[str],
#                         entity_ids:List[str], scores:List[float]
#                  ):
#         self.mention = mention
#         self.mention_context = mention_context
#         self.mention_position = mention_position
#         self.entity_cands = entity_cands
#         self.entity_context = entity_context
#         self.entity_ids = entity_ids
#         self.scores = scores


class DataSet():
    def __init__(self, dataset, transform):
        self.data = dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = self.transform(sample)
        return sample


class ToTensor():
    def __init__(self, entity_context_vocab, mention_context_vocab, entity_vocab):
        self.entity_context_vocab = entity_context_vocab
        self.mention_context_vocab = mention_context_vocab
        self.entity_vocab = entity_vocab

    def __call__(self, sample):
        mention : List = list(sample.mention)
        mention_context : List =  list(sample.mention_context)
        mention_position : List = sample.mention_position
        entity_cands : List = sample.entity_cands
        entity_contexts : List[List[str]] = [list(context) for context in sample.entity_context]

        mention_id = [self.mention_context_vocab.word2id(m) for m in mention]
        mention_context_id = [self.mention_context_vocab.word2id(m) for m in mention_context]
        entity_cands_id = [self.entity_vocab.word2id(m) for m in entity_cands]
        entity_contexts_id = [[self.entity_context_vocab.word2id(m[i]) for i in range(len(m))] for m in entity_contexts]

        return {'mention':mention_id,
                'mention_context':mention_context_id,
                'mention_position':mention_position,
                'entity_cands_id':entity_cands_id,
                'entity_contexts_id':entity_contexts_id
                }



def loading_dataset(token_vocab, label_vocab, bert=False):
    datas = pickle.load(open('data.pkl','rb'))
    np.random.shuffle(datas)
    train_data = datas[:int(len(datas)*0.8)]
    test_data = datas[int(len(datas)*0.8):]
    print(f'train_data len : {len(train_data)}, test_data len : {len(test_data)}')
    train_dataset = DataSet(train_data, transform=transforms.Compose([ToTensor(token_vocab, label_vocab)]))
    test_dataset = DataSet(test_data, transform=transforms.Compose([ToTensor(token_vocab, label_vocab)]))
    return train_dataset, test_dataset

