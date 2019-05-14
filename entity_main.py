from detect_mention_model import DetectModel
from entity_dataloader import  *
import torch.optim as optim
from torch.utils.data import DataLoader
from allennlp.nn.util import move_to_device
import os
from utils import collate_fn_entity_link
from entity_linking_model import EntityLink
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

'''
'mention':mention_id,
'mention_context':mention_context_id,
'mention_position':mention_position,
'entity_cands_id':entity_cands_id,
'entity_contexts_id':entity_contexts_id
'''

def get_accuracy(scores, target):
    '''
    :param scores: batch, cands
    :param target: batch
    :return:
    '''
    value, position = torch.max(scores, dim=-1)
    return torch.sum(position == target).float() / scores.size(0)

def train(model, train_dataset, test_dataset, opt):
    epoch = 0
    step = 6000
    while True:
        model.train()
        train_loader = DataLoader(train_dataset,num_workers=12, batch_size=32, shuffle=True, collate_fn=collate_fn_entity_link)
        valid_loader = DataLoader(test_dataset, num_workers=4, batch_size=64, collate_fn=collate_fn_entity_link)
        train_loss = []
        train_acc = []
        for batch in train_loader:
            batch = move_to_device(batch, 0)
            mention_context = batch['mention_context']
            mention_position = batch['mention_position']
            entity_cands = batch['entity_cands_id']
            entity_context = batch['entity_contexts_id']
            target = batch['target']
            opt.zero_grad()
            scores, loss = model(mention_context, mention_position, entity_context, entity_cands, target)
            train_acc.append(get_accuracy(scores, target))
            loss.backward()
            train_loss.append(loss.item())
            opt.step()
            step += 1
            if step % 1000 == 0:
                torch.save(model.state_dict(), os.path.join('entity_model', str(step) + '.pt'))
                print(f'epoch : {epoch}, loss : {sum(train_loss) / len(train_loss)}, accu : {sum(train_acc) / len(train_acc)}')

                model.eval()
                eval_loss = []
                eval_acc = []
                for batch in valid_loader:
                    batch = move_to_device(batch, 0)
                    mention_context = batch['mention_context']
                    mention_position = batch['mention_position']
                    entity_cands = batch['entity_cands_id']
                    entity_context = batch['entity_contexts_id']
                    target = batch['target']
                    scores, loss = model(mention_context, mention_position, entity_context, entity_cands, target)

                    eval_loss.append(loss.item())
                    eval_acc.append(get_accuracy(scores, target))
                print(f'eval loss : {sum(eval_loss) / len(eval_loss)}, accu : {sum(eval_acc) / len(eval_acc)}')
                #torch.save(model.state_dict(), os.path.join('entity_model', str(sum(eval_acc) / len(eval_acc))+'.pt'))
                model.train()
                train_loss = []
                train_acc = []
        epoch += 1


from tqdm import tqdm
def predict(model, dataset, output_path):
    res = []
    dataloader =  DataLoader(dataset,num_workers=1, batch_size=1, shuffle=False, collate_fn=collate_fn_entity_link)
    for batch in tqdm(dataloader):
        batch = move_to_device(batch, 0)
        mention_context = batch['mention_context']
        mention_position = batch['mention_position']
        entity_cands = batch['entity_cands_id']
        entity_context = batch['entity_contexts_id']
        mention = batch['mention']
        text_id = batch['text_id']
        #spilt entity_cands, entity_context
        entity_cands_list = entity_cands.split(30)
        entity_context_list = entity_context.split(30)
        assert len(entity_cands_list) == len(entity_context_list)
        mention_context.cuda(0)
        mention_position.cuda(0)
        s = 0
        id = -1
        for entity_cand, entity_con in zip(entity_cands_list, entity_context_list):
            entity_cand.cuda(0)
            entity_con.cuda(0)
            scores = model(mention_context, mention_position, entity_con, entity_cand)
            vals, ids = torch.max(scores, dim=-1)
            print(vals, ids)
        for i, (t_id, m, id) in enumerate(zip(text_id, mention, ids.item())):
            res.append({'text_id':t_id, 'mention':m, 'kb_id' : entity_cands[i][id]})
    if not os.path.exists('entity_link_valid_dataset'):
        os.mkdir('entity_link_valid_dataset')
    pkl.dump(res, open(os.path.join('entity_link_valid_dataset', output_path+'.pkl'), 'wb'))



from detect_mention_data import Vocab

if __name__ == '__main__':


    entity_context_vocab = Vocab('vocab/entity_context_vocab.txt')
    mention_context_vocab = Vocab('vocab/vocab.txt')
    entity_vocab = Vocab('vocab/entity_vocab.txt')
    print(f'entity_context_vocab size : {len(entity_context_vocab)}')
    print(f'mention_context_vocab size : {len(mention_context_vocab)}')
    print(f'entity_vocab size : {len(entity_vocab)}')


    #train_dataset, test_dataset = loading_dataset(entity_context_vocab, mention_context_vocab, entity_vocab)

    model = EntityLink(len(mention_context_vocab),
                       len(entity_context_vocab),
                       len(entity_vocab),
                       emb_dim=128,
                       hid_dim=256)

    model.to('cuda')
    #opt = optimization.BertAdam(model.parameters(), lr=5e-5)
    opt = optim.Adam(model.parameters())
    model.load_state_dict(torch.load('entity_model/6000.pt'))
    #train(model, train_dataset, test_dataset, opt)
    predict_dataset = loading_predict_dataset(entity_context_vocab, mention_context_vocab, entity_vocab)
    predict(model, predict_dataset, 'first')