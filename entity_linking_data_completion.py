from entity_link_data import Mention, compute_rouge_l
import pickle as pkl
from collections import OrderedDict
from multiprocess_generate_data import combine
from tqdm import tqdm
from multiprocessing import Process, cpu_count


def find_id_position(target, ls):
    for idx, item in enumerate(ls):
        if item[0] == target:
            return idx
    return -1


def search(dic, ls):
    entity_cands = []
    entity_context = []
    entity_ids = []
    for item in ls:
        v = dic[item[0]]
        entity_cands.append(v['subject'])
        entity_context.append(v['text'])
        entity_ids.append(item[0])
    return entity_cands, entity_context, entity_ids


def complementation(data_path, kb_dict_path, output_path, avg=None):
    datas = pkl.load(open(data_path, 'rb'))
    kb_dict = pkl.load(open(kb_dict_path, 'rb'))
    if avg is None:
        cands = []
        #first get the statistic information of candidates
        for data in datas:
            cands.append(len(data.entity_cands))
        print('cands max : {}, min : {}, avg : {}'.format(max(cands), min(cands), sum(cands) / len(cands)))
        # the avg number are set to cands number
        cands_num = int(sum(cands) / len(cands))
    else:
        cands_num = avg

    for data in tqdm(datas):
        target_id_pos = data.entity_ids.index(data.target_id)
        if len(data.entity_cands) > cands_num:
            #如果target出现在cands_num之内
            target_id = data.target_id
            if target_id_pos < cands_num:
                data.entity_cands = data.entity_cands[:cands_num]
                data.entity_context = data.entity_context[:cands_num]
                data.entity_ids = data.entity_ids[:cands_num]
                data.scores = data.scores[:cands_num]
                data.target_position = target_id_pos
            else:
                data.entity_cands = data.entity_cands[:cands_num-1] + [data.target_entity]
                data.entity_context = data.entity_context[:cands_num-1] + [data.entity_context[target_id_pos]]
                data.entity_ids = data.entity_ids[:cands_num-1] + [target_id]
                data.scores = data.scores[:cands_num-1] + [data.scores[target_id_pos]]
                data.target_position = cands_num

        elif len(data.entity_cands) == cands_num:
            data.target_position = target_id_pos

        else:
            mention = data.mention
            l = {}
            for kb_id, value in kb_dict.items():
                entity = value['subject']
                score = compute_rouge_l(mention, entity)
                l[kb_id] = score
            #sort
            l = sorted(l.items(), key=lambda item:item[-1], reverse=True)[:cands_num]
            num = find_id_position(data.target_id ,l)
            if num != -1:
                data.entity_cands, data.entity_context, data.entity_ids = search(kb_dict, l)
                data.scores = [item[-1] for item in l]
            else:
                target_entity = data.target_entity
                target_context = data.entity_context[data.entity_ids.index(data.target_id)]
                target_id = data.target_id
                target_score = data.scores[data.entity_ids.index(data.target_id)]

                data.entity_cands, data.entity_context, data.entity_ids = search(kb_dict, l[:-1])
                data.entity_cands += [target_entity]
                data.entity_context += [target_context]
                data.entity_ids += [target_id]

                data.scores = [item[-1] for item in l] + [target_score]
                data.target_position = cands_num

            assert len(data.scores) == len(data.entity_ids) == len(data.entity_context) == cands_num
    pkl.dump(datas, open(output_path,'wb'))



import os

#257078
if __name__ == '__main__':
    combine(data_dir=os.path.join('entity_link_dataset', 'entity_link_dataset'),
            save_path=os.path.join('entity_link_dataset','data.pkl'))
    # ps = []
    # for data_path in glob.glob(os.path.join('e_d_r', '*')):
    #     output = data_path.split('/')[-1]
    #     ps.append(Process(target=complementation, args=(data_path,os.path.join('kb_info','kb_dict.pkl'),
    #                                           os.path.join('entity_link_dataset',output), 30, )))
    # for p in ps:
    #     print('p : {} start'.format(p.name))
    #     p.start()
    # for p in ps:
    #     p.join()