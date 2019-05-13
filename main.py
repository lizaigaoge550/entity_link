from detect_mention_model import DetectModel
from detect_mention_data import  *
import torch.optim as optim
from torch.utils.data import DataLoader
from allennlp.nn.util import move_to_device
import os
from pytorch_pretrained_bert import optimization
import torch
from collections import Counter


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def extract_mention(tags, tokens):
    res = []
    i = 0
    while i < len(tags):
        if tags[i] == 'B':
            j = i+1
            while j < len(tags):
                if tags[j] == 'I':
                    j += 1
                else:
                    break
            if j < len(tags):
                if tags[j] == 'E':
                    res.append((i, ''.join(tokens[i:j+1])))
            i = j
        elif tags[i] == 'S':
            res.append((i, tokens[i]))
            i += 1
        else:
            i += 1
    return res

def compute_f(predicted_tags, tags, label_vocab, original_tokens, e):
    output = []
    assert len(tags) == len(predicted_tags)
    f_values = []
    for p_tag, t_tag, tokens in zip(predicted_tags, tags, original_tokens):
        t_tag = t_tag[:len(p_tag)]
        tokens = tokens[:len(p_tag)]
        t_tag = [label_vocab.id2word(t.item()) for t in t_tag]
        p_tag = [label_vocab.id2word(p) for p in p_tag]
        output.append({'tokens': tokens, 't_tag': t_tag, 'p_tag':p_tag})
        true_mention = extract_mention(t_tag, tokens)
        predict_mention = extract_mention(p_tag, tokens)
        common = 0
        for t_mention in true_mention:
            for p_mention in predict_mention:
                if t_mention[0] == p_mention[0] and t_mention[1] == p_mention[1]:
                    common += 1
                    break
                elif p_mention[0] > t_mention[0]:
                    break

        if len(true_mention) == 0 or len(predict_mention) == 0:
            f_values.append(int(len(predict_mention) == len(true_mention)))
            #recall = 0
            #print(t_tag)
            continue
        else:
            recall = common / len(true_mention)
            #f_values.append(recall)
            precision = common / len(predict_mention)
            f_values.append((2*recall*precision) / (recall + precision + 1e-6))
    avg_f = sum(f_values) / len(f_values)
    #pickle.dump(output, open(os.path.join('valid_log', str(e)+'_'+str(avg_f)+'.pkl'), 'wb'))
    return f_values, output

def train(model, train_dataset, test_dataset, opt, label_vocab):
    epoch = 0
    while True:
        model.train()
        train_loader = DataLoader(train_dataset,num_workers=4, batch_size=64, shuffle=True)
        valid_loader = DataLoader(test_dataset, num_workers=4, batch_size=32)
        train_loss = []
        train_f = []
        for batch in train_loader:
            batch = move_to_device(batch, 0)
            tokens = batch['tokens']
            tags = batch['tags']
            original_tokens = batch['original_tokens']
            opt.zero_grad()
            loss, best_paths, token_lens = model(tokens, tags)
            loss = -loss
            loss.backward()
            train_loss.append(loss.item())
            opt.step()
            predicted_tags = [x for x, y in best_paths]
            #f1 = compute_f(predicted_tags, tags, label_vocab, original_tokens, epoch)
            #train_f += f1
        print(f'epoch : {epoch}, loss : {sum(train_loss) / len(train_loss)}')
        epoch += 1
        model.eval()
        valid_loss = []
        valid_f = []
        #valid_output = []
        for batch in valid_loader:
            batch = move_to_device(batch, 0)
            tokens = batch['tokens']
            tags = batch['tags']
            original_tokens = batch['original_tokens']
            opt.zero_grad()
            loss, best_paths, _ = model(tokens, tags)
            loss = -loss
            valid_loss.append(loss.item())
            predict = [x for x, y in best_paths]
            f, output = compute_f(predict, tags, label_vocab, original_tokens, epoch)
            valid_f += f
            #valid_output += output
        # pickle.dump(valid_output, open(os.path.join('valid_log_5e5', str(epoch)+'.pkl'),'wb'))
        if not os.path.exists('detect_model_normal_0.001'):
            os.mkdir('detect_model_normal_0.001')
        torch.save(model.state_dict(), os.path.join('detect_model_normal_0.001', str(sum(valid_f) / len(valid_f)) + '.pt'))
        print(f'valid Loss : {sum(valid_loss) / len(valid_loss)}, valid f : {sum(valid_f) / len(valid_f)}')

from utils import collate_fn
def train_bert(model, train_dataset, test_dataset, opt, label_vocab):
    epoch = 0
    while True:
        model.train()
        train_loss = []
        train_loader = DataLoader(train_dataset,num_workers=4, batch_size=64, collate_fn=collate_fn, shuffle=True)
        valid_loader = DataLoader(test_dataset, num_workers=4, batch_size=32, collate_fn=collate_fn)
        for batch in train_loader:
            batch = move_to_device(batch, 0)
            tokens = batch['tokens']
            tags = batch['tags']
            offset = batch['offset']
            opt.zero_grad()
            loss = model.bert_forward(tokens, offset, tags, is_training=True)
            loss = -loss
            loss.backward()
            train_loss.append(loss.item())
            opt.step()
        print(f'epoch : {epoch}, loss : {sum(train_loss) / len(train_loss)}')
        epoch += 1
        model.eval()
        valid_loss = []
        valid_f = []
        valid_output = []
        for batch in valid_loader:
            batch = move_to_device(batch, 0)
            tokens = batch['tokens']
            tags = batch['tags']
            offset = batch['offset']
            original_tokens = batch['original_tokens']
            opt.zero_grad()
            loss, best_paths = model.bert_forward(tokens, offset, tags, is_training=False)
            loss = -loss
            valid_loss.append(loss.item())
            predict = [x for x, y in best_paths]
            f, output = compute_f(predict, tags, label_vocab, original_tokens, epoch)
            valid_f += f
            valid_output += output
        #pickle.dump(valid_output, open(os.path.join('valid_log_5e5', str(epoch)+'.pkl'),'wb'))
        if not os.path.exists('detect_model_0.001'):
            os.mkdir('detect_model_0.001')
        torch.save(model.state_dict(), os.path.join('detect_model_0.001', str(sum(valid_f) / len(valid_f))+'.pt'))
        print(f'valid Loss : {sum(valid_loss) / len(valid_loss)}, valid f : {sum(valid_f) / len(valid_f)}')


'''
"text_id"
"text", 
"mention_data": [{"mention": "南京南站", "offset": "0"},....]
'''
def generate_entity(model, test_dataset, label_vocab, model_name):
    res = []
    model.eval()
    test_loader = DataLoader(test_dataset, num_workers=4, batch_size=32, collate_fn=collate_fn, drop_last=False)
    for batch in tqdm(test_loader):
        batch = move_to_device(batch, 0)
        text_ids = batch['text_ids']
        tokens = batch['tokens']
        offset = batch['offset']
        original_tokens = batch['original_tokens']
        best_paths = model.bert_forward(tokens, offset, tags=None, is_training=False)
        predict = [x for x, y in best_paths]
        p_tags = [[label_vocab.id2word(p) for p in p_tag] for p_tag in predict]
        predict_mentions = [extract_mention(p_tag, original_tokens[i]) for i, p_tag in enumerate(p_tags)]
        predict_mentions = [[{'mention':item[-1], 'offset':item[0]} for item in predict_mention] for predict_mention in predict_mentions]
        for text_id, text, metion_data in zip(text_ids, original_tokens, predict_mentions):
           res.append({'text_id':text_id, 'text':text, 'mention_data':metion_data})
    assert len(res) == len(test_dataset)
    if not os.path.exists('valid_detect_mention'):
        os.mkdir('valid_detect_mention')
    pickle.dump(res, open(os.path.join('valid_detect_mention', model_name+'.pkl'),'wb'))



def get_top(mentions):
    d = {}
    for mention in mentions:
        key = str(mention['offset']) + '@' + mention['mention']
        if key in d:
            d[key] += 1
        else:
            d[key] = 1

    res = []
    # for item in Counter(d).most_common():
    #     key, value = item[0], item[1]
    #     if value >= len(mentions) // 2:
    #        res.append({'mention':key.split('@')[-1], 'offset':key.split('@')[0]})
    if res == []:
        #get_top_3

        for item in Counter(d).most_common():
            if len(res) >= len(d) // 2:
                break
            key, value = item[0], item[1]
            res.append({'mention': key.split('@')[-1], 'offset': key.split('@')[0]})
    return res


def result_ensemble(data_dir):
    pre_data = {}
    for data_path in glob.glob(os.path.join(data_dir, '*')):
        print(data_path)
        data = pkl.load(open(data_path,'rb'))
        for item in data:
            if item['text_id'] not in pre_data:
                pre_data[item['text_id']] = {'text':item['text'], 'mention_data':item['mention_data']}
            else:
                pre_data[item['text_id']]['mention_data'] += item['mention_data']
    for key, value in pre_data.items():
        mentions = value['mention_data']
        #statistic
        value['mention_data'] = get_top(mentions)
    pkl.dump(pre_data, open('valid_detect_mention/ensemble.pkl','wb'))


import glob
if __name__ == '__main__':
    result_ensemble('valid_detect_mention')
    # token_vocab = Vocab('checkpoint/vocab.txt')
    # label_vocab = Vocab('vocab/label_vocab.txt')
    # token_vocab_size = len(token_vocab)
    # label_vocab_size = len(label_vocab)
    # print(f'token_vocab size : {len(token_vocab)}')
    # train_dataset, test_dataset = loading_dataset(token_vocab, label_vocab)
    # develop_dataset = loading_develop_dataset(token_vocab, label_vocab)
    #
    # model = DetectModel(vocab_size=token_vocab_size, input_dim=768,
    #                     output_dim=256, num_tags=label_vocab_size,
    #                     n_layer=2,
    #                     constrains=[(0,0),(1,0),(1,2),(1,6),
    #                                 (2,3),(2,4),(3,3),(3,4),(4,0),
    #                                 (4,1),(4,6), (6,0)
    #                                 ]
    #                     )
    # model.to('cuda')
    # #opt = optimization.BertAdam(model.parameters())
    # opt = optim.Adam(model.parameters())
    # for model_path in glob.glob('detect_model_0.001/*'):
    #     model.load_state_dict(torch.load(model_path))
    #     #train(model, train_dataset, test_dataset, opt, label_vocab)
    #     print(f'{model_path} data is generating...')
    #     a = model_path.split('/')[-1]
    #     generate_entity(model, develop_dataset, label_vocab, str(a))