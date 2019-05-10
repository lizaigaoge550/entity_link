from multiprocessing import Process
from multiprocessing import cpu_count
import pickle
from pytorch_pretrained_bert import BasicTokenizer
import os
import bisect
import json
import numpy as np
import glob

'。.^—-_'

def update_offset(offset, space):
    offset -= bisect.bisect_left(space, offset)
    return offset

def is_punc(text):
    for i in range(len(text)):
        if  text[i] not in '★゜╟[].!//_,$&%^*()<>+\"\'?@#-|:~{}——！\\\\，。=？、：“”‘’《》【】￥……（）╬':
            return False
    return True

def mention_contain_punc(mention):
    return mention[-1] in '★゜╟[].!//_,$&%^*()<>+\"\'?@#-|:~{}——！\\\\，。=？、：“”‘’《》【】￥……（）'



def remove_end_punction(text):
    for i in range(len(text)-1,-1,-1):
        if text[i] in '。.^—-_':
            continue
        else:
            return text[:i+1]
    return text[:i]

def remove_no_chinese(tokenizer, text, is_str=False):
    space = []
    r_text = []
    for i in range(len(text)):
        #if text[i] in '★゜╟[].!//_,$&%^*()<>+\"\'?@#-|:~{}——！\\\\，。=？、：“”‘’《》【】￥……（）╬':
        #    space.append(i)
        #    continue
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
        data['text'] = remove_end_punction(data['text'])
        text_len = len(data['text'])
        text, space = remove_no_chinese(BasicTokenizer(), data['text'])
        assert len(text) == text_len - len(space)
        tags = ['O' for _ in range(len(text))]
        assert len(text) == len(tags)
        mentionlist = data['mention_data']
        new_mentionlist = []
        for mention in mentionlist:
            #{"kb_id": "311223", "mention": "南京南站", "offset": "0"}
            mention_str = remove_end_punction(mention['mention'])
            if mention_str:
                mention_str = remove_no_chinese(BasicTokenizer(), mention_str, is_str=True)
                new_mentionlist.append(mention_str)
                offset = int(mention['offset'])
                offset = update_offset(offset, space)
                if len(mention_str) > 1:
                    tags[offset:offset+len(mention_str)] = ['B'] + ['I'] * (len(mention_str)-2) + ['E']
            #else:
            #    tags[offset] = 'S'

        assert len(text) == len(tags), f'{text}, {tags}'
        generate_data.append({'tokens':text, 'tags':tags})
        if origin_len != text_len and len(new_mentionlist) != len(mentionlist):
            print(f'tokens : {text}, tag : {tags}, mention : {[m["mention"] for m in mentionlist]}, new_mention : {new_mentionlist}')
        count += 1
    pickle.dump(generate_data, open(os.path.join('pkl',f'{str(i)}.pkl'),'wb'))


def combine(data_dir):
    data = []
    for file_name in glob.glob(os.path.join(data_dir, '*')):
        data += pickle.load(open(file_name,'rb'))
    print(len(data))
    pickle.dump(data, open('data.pkl','wb'))


def check_mention():
    datas = [line for line in open(os.path.join('ccks2019_el', 'train.json'), encoding='utf-8').readlines()]
    for data in datas:
        data = json.loads(data)
        mentionlist = data['mention_data']
        mention = mentionlist[-1]
        mention_str = mention['mention']
        offset = mention['offset']
        text = data['text']
        if is_punc(text[int(offset)+len(mention_str):]):
            if mention_contain_punc(mention_str):
                print(mention_str)

if __name__ == '__main__':
    #check_mention()
    # for data in pickle.load(open('data.pkl','rb')):
    #   print(data)
    #combine('pkl')
    # datas = [line for line in open(os.path.join('ccks2019_el', 'train.json'), encoding='utf-8').readlines()]
    # datalist = np.array_split(datas, cpu_count())
    # print(len(datalist))
    # ps = []
    # i = 0
    # for data in datalist:
    #     ps.append(Process(target=generate_data, args=(data, i,)))
    #     i += 1
    # for p in ps:
    #     print(f'{p.name} start.......')
    #     p.start()
    # for p in ps:
    #     p.join()
    s = []
    datas = [line for line in open(os.path.join('ccks2019_el', 'train.json'), encoding='utf-8').readlines()]
    for data in datas:
        data_json = json.loads(data)
        mention_data = data_json['mention_data']
        for mention in mention_data:
            mention_str = mention['mention']
            mention_str = remove_end_punction(mention_str)
            if mention_str == '':
                s.append(data)
                break
    print(s)
    fw = open('5-9.json', 'a', encoding='utf-8')
    for i in range(len(s)):
        json.dump(s[i], fw)
        fw.write('\n')