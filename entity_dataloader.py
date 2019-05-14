import pickle
import numpy as np
from torchvision import transforms
from entity_link_data import Mention
from typing import List
from detect_mention_data import Vocab
import pickle as pkl

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
        entity_cands : List = sample.entity_ids
        entity_contexts : List[List[str]] = [list(context) for context in sample.entity_context]
        target_id  = sample.target_id
        target_position = entity_cands.index(target_id)

        mention_id = [self.mention_context_vocab.word2id(m) for m in mention]
        mention_context_id = [self.mention_context_vocab.word2id(m) for m in mention_context]
        entity_cands_id = [self.entity_vocab.word2id(m, type='label') for m in entity_cands]
        entity_contexts_id = [[self.entity_context_vocab.word2id(m[i]) for i in range(len(m))] for m in entity_contexts]

        return {'mention':mention_id,
                'mention_context':mention_context_id,
                'mention_position':mention_position,
                'entity_cands_id':entity_cands_id,
                'entity_contexts_id':entity_contexts_id,
                'target_id':target_id,
                'target_position':target_position
                }


class TestToTensor():
    def __init__(self, entity_context_vocab, mention_context_vocab, entity_vocab):
        self.entity_context_vocab = entity_context_vocab
        self.mention_context_vocab = mention_context_vocab
        self.entity_vocab = entity_vocab

    def __call__(self, sample):
        mention : str = sample['mention']
        mention_context : List = list(sample['mention_text'])
        mention_position : List = sample['mention_position']
        mention_position : List = list(map(lambda a:int(a), mention_position))
        entity_cands : List = sample['entity_ids']
        entity_contexts : List[List[str]] = [list(context) for context in sample['entity_text']]

        mention_context_id = [self.mention_context_vocab.word2id(m) for m in mention_context]
        entity_cands_id = [self.entity_vocab.word2id(m, type='label') for m in entity_cands]
        entity_contexts_id = [[self.entity_context_vocab.word2id(m[i]) for i in range(len(m))] for m in entity_contexts]

        return {'text_id':sample['text_id'],
                'mention':mention,
                'mention_context':mention_context_id,
                'mention_position':mention_position,
                'entity_cands_id':entity_cands_id,
                'entity_contexts_id':entity_contexts_id,
                }

def loading_dataset(entity_context_vocab, mention_context_vocab, entity_vocab):
    datas = pickle.load(open('entity_link_dataset/data.pkl','rb'))
    np.random.shuffle(datas)
    train_data = datas[:int(len(datas)*0.8)]
    test_data = datas[int(len(datas)*0.8):]
    print(f'train_data len : {len(train_data)}, test_data len : {len(test_data)}')
    train_dataset = DataSet(train_data, transform=transforms.Compose([ToTensor(entity_context_vocab, mention_context_vocab, entity_vocab)]))
    test_dataset = DataSet(test_data, transform=transforms.Compose([ToTensor(entity_context_vocab, mention_context_vocab, entity_vocab)]))
    return train_dataset, test_dataset

def loading_predict_dataset(entity_context_vocab, mention_context_vocab, entity_vocab):
    datas = pickle.load(open('entity_link_dataset/valid_entity_linking.pkl','rb'))
    print(f'data len : {len(datas)}')
    dataset = DataSet(datas, transform=transforms.Compose([TestToTensor(entity_context_vocab, mention_context_vocab, entity_vocab)]))
    return dataset


if __name__ == '__main__':
    entity_context_vocab = Vocab('vocab/entity_context_vocab.txt')
    mention_context_vocab = Vocab('vocab/vocab.txt')
    entity_vocab = Vocab('vocab/entity_vocab.txt')
    datas = pkl.load(open('entity_link_dataset/valid_entity_linking.pkl','rb'))
    for d in DataSet(datas, transform=transforms.Compose([TestToTensor(entity_context_vocab, mention_context_vocab, entity_vocab)])):
        print(d)