import pickle
import os
import json
from collections import Counter
import bisect
from torchvision import transforms
from tqdm import tqdm
from pytorch_pretrained_bert.tokenization import BasicTokenizer, BertTokenizer
from utils import *

def update_offset(offset, space):
    offset -= bisect.bisect_left(space, offset)
    return offset


def remove_no_chinese(tokenizer, text, is_str=False):
    space = []
    r_text = []
    for i in range(len(text)):
        if len(tokenizer.tokenize(text[i])):
            r_text.append(text[i])
        else:
            space.append(i)
    if is_str == False:
        return r_text ,space
    else:
        return ''.join(r_text)


def generate_data():
    count = 0
    generate_data = []
    datas = [line for line in open(os.path.join('ccks2019_el','train.json'), encoding='utf-8').readlines()]
    for data in tqdm(datas):
        data = json.loads(data)
        origin_len = len(data['text'])
        text, space = remove_no_chinese(BasicTokenizer(), data['text'])
        assert len(text) == origin_len - len(space)
        tags = ['O' for _ in range(len(text))]
        assert len(text) == len(tags)
        mentionlist = data['mention_data']
        for mention in mentionlist:
            if mention['kb_id'] == 'NIL':
                continue
            #{"kb_id": "311223", "mention": "南京南站", "offset": "0"}
            mention_str = remove_no_chinese(BasicTokenizer(), mention['mention'], is_str=True)
            offset = int(mention['offset'])
            offset = update_offset(offset, space)
            if len(mention_str) > 1:
                tags[offset:offset+len(mention_str)] = ['B'] + ['I'] * (len(mention_str)-2) + ['E']
            #else:
            #    tags[offset] = 'S'

        assert len(text) == len(tags), f'{text}, {tags}'
        generate_data.append({'tokens':text, 'tags':tags})
        count += 1
    pickle.dump(generate_data, open('data.pkl','wb'))


def build_vocab():
    vocab_dict = {}
    datas = pickle.load(open('data.pkl','rb'))
    for data in datas:
        tokens = data['tokens']
        for token in tokens:
            if token not in vocab_dict:
                vocab_dict[token] = 1
            else:
                vocab_dict[token] += 1

    fw = open('vocab.txt','w',encoding='utf-8')
    fw.write('[PAD]\n')
    fw.write('[UNK]\n')
    for i, item in enumerate(Counter(vocab_dict).most_common()):
            fw.write(item[0]+'\n') if i < len(vocab_dict)-1 else fw.write(item[0])
    fw.close()


def data_statistic():
    datas = pickle.load(open('data.pkl','rb'))
    ls = [len(data['tokens']) for data in datas]
    print(f'max len: {max(ls)}, min len : {min(ls)}, avg len : {sum(ls) / len(ls)}')
    return ls


class Vocab():
    def __init__(self, vocab_file):
        self.word_to_idx = {}
        self.id_to_word = {}
        with open(vocab_file, 'r') as fr:
            for line in fr.readlines():
                word = line.split('\n')[0]
                if word in self.word_to_idx:
                    raise ("Duplicate word : {} exist".format(word))
                self.word_to_idx[word] = len(self.word_to_idx)
                self.id_to_word[len(self.word_to_idx)-1] = word

    def word2id(self, word, type='token'):
        if type == 'token':
            return self.word_to_idx.get(word, self.word_to_idx['[UNK]'])
        else:
            return self.word_to_idx[word]
    def id2word(self, id):
        return self.id_to_word[id]

    def __len__(self):
        return len(self.word_to_idx)


def loading_dataset(token_vocab, label_vocab, bert=False):
    datas = pickle.load(open('detect_mention_dataset/data.pkl','rb'))
    np.random.shuffle(datas)
    train_data = datas[:int(len(datas)*0.8)]
    test_data = datas[int(len(datas)*0.8):]
    print(f'train_data len : {len(train_data)}, test_data len : {len(test_data)}')
    if bert == False:
        train_dataset = DataSet(train_data, transform=transforms.Compose([ToTensor(token_vocab, label_vocab)]))
        test_dataset = DataSet(test_data, transform=transforms.Compose([ToTensor(token_vocab, label_vocab)]))
    else:
        train_dataset = DataSet(train_data, transform=transforms.Compose([BertTensor(token_vocab, label_vocab)]))
        test_dataset = DataSet(test_data, transform=transforms.Compose([BertTensor(token_vocab, label_vocab)]))
    return train_dataset, test_dataset


def loading_develop_dataset(token_vocab, label_vocab):
    datas = pickle.load(open('detect_mention_dataset/develop.pkl', 'rb'))
    dataset = DataSet(datas, transform=transforms.Compose([BertTensor(token_vocab, label_vocab, is_training=False)]))
    return dataset


class DataSet(object):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = self.transform(sample)
        return sample


class ToTensor(object):
    def __init__(self, token_vocab, label_vocab, **kwargs):
        self.kwargs = kwargs
        self.token_vocab = token_vocab
        self.label_vocab = label_vocab

    def __call__(self, sample):
        max_len = self.kwargs.get('max_token_lens', 50)
        tokens = [self.token_vocab.word2id(token) for token in sample['tokens']]
        tags = [self.label_vocab.word2id(tag, type='label') for tag in sample['tags']]

        assert len(tokens) == len(tags)
        #padding
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
            tags = tags[:max_len]
        elif len(tokens) < max_len:
            tokens = tokens + (max_len - len(tokens)) * [0]
            tags = tags + (max_len - len(tags)) * [0]

        assert len(tokens) == max_len
        assert len(tags) == max_len
        return {'tokens': torch.LongTensor(tokens), 'tags':torch.LongTensor(tags), 'original_tokens':sample['tokens']}


class BertTensor(object):
    def __init__(self, token_vocab, label_vocab, is_training=True, **kwargs):
        self.kwargs = kwargs
        self.token_vocab = token_vocab
        self.label_vocab = label_vocab
        self.tokenizer = BertTokenizer('checkpoint/vocab.txt')
        self.is_training = is_training

    def chunk(self, word_idx, offset, max_seq_len):
        while len(offset) > max_seq_len:
            idx = offset.pop()
        return word_idx[:idx], offset

    def __call__(self, sample):
        max_len = self.kwargs.get('max_token_lens', 50)
        word_idx, offset = tokens_to_indices(sample['tokens'], vocab=self.token_vocab, tokenizer=self.tokenizer,
                                             max_pieces=max_len)
        assert len(offset) == len(set(offset)), '{}, {}'.format(sample['tokens'], offset)
        if self.is_training:
            tags = [self.label_vocab.word2id(tag, type='label') for tag in sample['tags']]
            #
            if len(tags) > len(offset):
                tags = tags[:len(offset)]
            assert len(offset) == len(tags), f'{offset}, {tags}'
            return {'tokens': word_idx, 'tags':tags, 'offset': offset, 'origin_tokens':sample['tokens'][:len(offset)],
                'text_id':sample['text_id']}
        else:
            return {'tokens': word_idx, 'offset': offset, 'origin_tokens': sample['tokens'][:len(offset)],
                    'text_id': sample['text_id']}

#max len: 47, min len : 7, avg len : 21.035833333333333
#seq_len 47
if __name__ == '__main__':
    #generate_data()
    token_vocab = Vocab('vocab/vocab.txt')
    label_vocab = Vocab('vocab/label_vocab.txt')
    token_vocab_size = len(token_vocab)
    label_vocab_size = len(label_vocab)
    # print(f'token_vocab size : {len(token_vocab)}')
    train_dataset, test_dataset = loading_dataset(token_vocab, label_vocab, bert=True)
    for data in train_dataset:
        print(data)
