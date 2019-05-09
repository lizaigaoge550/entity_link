from multiprocessing import Process
from multiprocessing import cpu_count
import pickle
from pytorch_pretrained_bert import BasicTokenizer
import os
import bisect
import json
import numpy as np
import glob

def update_offset(offset, space):
    offset -= bisect.bisect_left(space, offset)
    return offset


def remove_end_punction(text):
    for i in range(len(text)-1,-1,-1):
        if text[i] < '\u4e00' or text[i] > '\u9fff':
            continue
        else:
            break
    return text[:i+1]

def remove_no_chinese(tokenizer, text, is_str=False):
    space = []
    r_text = []
    for i in range(len(text)):
        if text[i] in '.!//_,$&%^*()<>+\"\'?@#-|:~{}——！\\\\，。=？、：“”‘’《》【】￥……（）':
            space.append(i)
            continue
        if len(tokenizer.tokenize(text[i])):
            r_text.append(text[i])
        else:
            space.append(i)
    if is_str == False:
        return r_text ,space
    else:
        return ''.join(r_text)


def generate_data(datas, i):
    count = 0
    generate_data = []
    for data in datas:
        data = json.loads(data)
        origin_len = len(data['text'])
        #first remove the punction in the end of origin len
        #data['text'] = remove_end_punction(data['text'])
        text, space = remove_no_chinese(BasicTokenizer(), data['text'])
        assert len(text) == origin_len - len(space)
        tags = ['O' for _ in range(len(text))]
        assert len(text) == len(tags)
        mentionlist = data['mention_data']
        for mention in mentionlist:
            #{"kb_id": "311223", "mention": "南京南站", "offset": "0"}
            mention_str = remove_no_chinese(BasicTokenizer(), mention['mention'], is_str=True)
            offset = int(mention['offset'])
            offset = update_offset(offset, space)
            if len(mention_str) > 1:
                tags[offset:offset+len(mention_str)] = ['B'] + ['I'] * (len(mention_str)-2) + ['E']
            else:
                tags[offset] = 'S'

        assert len(text) == len(tags), f'{text}, {tags}'
        generate_data.append({'tokens':text, 'tags':tags})
        count += 1
    pickle.dump(generate_data, open(os.path.join('pkl',f'{str(i)}.pkl'),'wb'))


def combine(data_dir):
    data = []
    for file_name in glob.glob(os.path.join(data_dir, '*')):
        data += pickle.load(open(file_name,'rb'))
    print(len(data))
    pickle.dump(data, open('data.pkl','wb'))


if __name__ == '__main__':
    #combine('pkl')
    datas = [line for line in open(os.path.join('ccks2019_el', 'train.json'), encoding='utf-8').readlines()]
    datalist = np.array_split(datas, cpu_count())
    print(len(datalist))
    ps = []
    i = 0
    for data in datalist:
        ps.append(Process(target=generate_data, args=(data, i,)))
        i += 1
    for p in ps:
        print(f'{p.name} start.......')
        p.start()
    for p in ps:
        p.join()