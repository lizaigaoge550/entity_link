import json
import os
from typing import List
from tqdm import tqdm
from collections import OrderedDict
import pickle as pkl
from multiprocessing import Process
from multiprocessing import cpu_count



class Mention(object):
    def __init__(self, mention:str, mention_context:str, mention_position:List[int],
                        entity_cands:List[str], entity_context:List[str], entity_position:List[List[int]],
                        entity_ids:List[str], scores:List[float]
                 ):
        self.mention = mention
        self.mention_context = mention_context
        self.mention_position = mention_position
        self.entity_cands = entity_cands
        self.entity_context = entity_context
        self.entity_position = entity_position
        self.entity_ids = entity_ids
        self.scores = scores


def analysis_kb():
    def get_text(data, subject):
        if len(data) == 0:
            return subject
        for i in range(len(data)):
            if data[i]['predicate'] == '摘要':
                return data[i]['object']
        for i in range(len(data)):
            if data[i]['predicate'] == '义项描述':
                return subject+'是'+data[i]['object']
        max_len = 0
        max_text = ''
        #找最长的描述
        for i in range(len(data)):
            if len(data[i]['predicate']) > max_len:
                max_len = len(data[i]['predicate'])
                max_text = data[i]['object']
        return subject + '是' + max_text
        #s.append((subject, data))
        #raise Exception('No summarization')

    kb_dict = {}
    for kb_data in kb_datas:
        kb_data = json.loads(kb_data)
        subject_id = kb_data['subject_id']
        if subject_id in kb_dict:
            raise Exception('key : {} exist'.format(subject_id))
        text = get_text(kb_data['data'], kb_data['subject'])
        start_pos = text.index(kb_data['subject'])
        kb_dict[subject_id] = {'type':kb_data['type'], 'subject':kb_data['subject'],
                               'text': text, 'position':[start_pos, start_pos+len(kb_data['subject'])-1]}
    return kb_dict


def _lcs_dp(a, b):
    dp = [[0 for _ in range(0, len(b)+1)]
          for _ in range(0, len(a)+1)]
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp


def _lcs_len(a, b):
    dp = _lcs_dp(a, b)
    return dp[-1][-1]


def compute_rouge_l(output, reference, mode='f'):
    assert mode in list('fpr')  # F-1, precision, recall
    lcs = _lcs_len(output, reference)
    if lcs == 0:
        score = 0.0
    else:
        precision = lcs / len(output)
        recall = lcs / len(reference)
        f_score = 2 * (precision * recall) / (precision + recall)
        score = f_score
    return float(score)



def get_cands(mention, kb_dict):
    c_l = {}
    max_score = 0
    max_subject = ''
    max_id = -1
    for key, value in kb_dict.items():
        subject = value['subject']
        score = compute_rouge_l(mention, subject)
        if score > 0.7:
            c_l[key] =  (score, subject, value['text'], value['position'])
            if score > max_score:
                max_score = score
                max_subject = subject
                max_id = key
    c_l = OrderedDict(sorted(c_l.items(), key=lambda item:item[-1][0], reverse=True))
    return c_l, max_score, max_subject, max_id



def generate_entity_linking_data(datas, file_name):
    p = []
    res = []
    exist = 0
    no_exist = 0
    kb_dict = analysis_kb()
    for data in tqdm(datas):
        data = json.loads(data)
        text =data['text']
        mention_data =data['mention_data']
        for m_d in mention_data:
            kb_id = m_d['kb_id']
            mention = m_d['mention']
            offset = m_d['offset']
            if kb_id in kb_dict:
                gold_entity = kb_dict[kb_id]['subject']
                #compute cands
                entity_cands, max_score, max_subject, max_id = get_cands(mention, kb_dict)
                if kb_id in entity_cands:
                    p.append({'mention': mention, 'candidates': entity_cands,
                              'candidates_len':len(entity_cands), 'gold_entity': gold_entity,
                              'score':entity_cands[kb_id][0], 'max_score':max_score, 'max_subject':max_subject, 'max_id':max_id})
                    exist += 1
                else:
                    entity_cands[kb_id] = (0, gold_entity)
                    print('mention : {}, gold_entity : {}, ROUGE_score : {}'.format(mention, gold_entity, compute_rouge_l(mention, gold_entity)) )
                    no_exist += 1

                entity_cands_str = [value[1] for key, value in entity_cands.items()]
                entity_cands_score = [value[0] for key, value in entity_cands.items()]
                entity_ids = list(entity_cands.keys())
                entity_context = [value[2] for key, value in entity_cands.items()]
                entity_position = [value[-1] for key, value in entity_cands.items()]
                res.append(Mention(mention=mention, mention_context=text, mention_position=[offset, offset+len(mention)-1],
                        entity_cands=entity_cands_str, entity_context=entity_context, entity_position=entity_position,
                        entity_ids=entity_ids,scores = entity_cands_score
                        ))
    print('f : {} exist : {}. no_exist : {}'.format(file_name, exist, no_exist))
    pkl.dump(p, open(os.path.join('e_d_e', str(file_name)+'e_l.pkl'),'wb'))
    pkl.dump(res, open(os.path.join('e_d_r', str(file_name) + 'r_l.pkl'), 'wb'))


import numpy as np
if __name__ == '__main__':
    kb_datas = [line for line in open(os.path.join('ccks2019_el', 'kb_data'), encoding='utf-8').readlines()]
    datas = [line for line in open(os.path.join('ccks2019_el', 'train.json'), encoding='utf-8').readlines()]
    datalist = np.array_split(datas, cpu_count())
    print(len(datalist))
    ps = []
    i = 0
    for data in datalist:
        ps.append(Process(target=generate_entity_linking_data, args=(data, i,)))
        i += 1
    for p in ps:
        print(f'{p.name} start.......')
        p.start()
    for p in ps:
        p.join()
