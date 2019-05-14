import pickle as pkl
import os
import json

def result_submit(data_path):
    #'text_id':t_id, 'mention':m, 'kb_id' : entity_cands[i][id]
    datas = pkl.load(open(data_path, 'rb'))
    d = {}
    for data in datas:
        text_id = data['text_id']
        mention = data['mention']
        kb_id =  data['kb_id']
        if text_id in d:
            d[text_id].append({'mention':mention, 'kb_id':kb_id})
        else:
            d[text_id] = [{'mention':mention, 'kb_id':kb_id}]
    #get_offset
    develop_datas = [line for line in open(os.path.join('ccks2019_el', 'develop.json'), encoding='utf-8').readlines()]
    for d_d in develop_datas:
        d_d = json.loads(d_d)
        text_id = d_d['text_id']
        text = d_d['text']
        mention_infos = d[text_id]
        mention_d = []
        for mention_info in mention_infos:
            mention = mention_info['mention']
            offset = text.index(mention)
            kb_id = mention_info['kb_id']
            mention_d.append({'mention':mention, 'offset':str(offset), 'kb_id':kb_id})
        d_d['mention_data'] = mention_d