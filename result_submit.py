import pickle as pkl

def result_submit(origin_data, data_path):
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