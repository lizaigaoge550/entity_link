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
    while True:
        model.train()
        train_loader = DataLoader(train_dataset,num_workers=4, batch_size=64, shuffle=True, collate_fn=collate_fn_entity_link)
        valid_loader = DataLoader(test_dataset, num_workers=4, batch_size=32, collate_fn=collate_fn_entity_link)
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
        print(f'epoch : {epoch}, loss : {sum(train_loss) / len(train_loss)}, accu : {sum(train_acc) / len(train_acc)}')
        epoch += 1




from detect_mention_data import Vocab

if __name__ == '__main__':


    entity_context_vocab = Vocab('vocab/entity_context_vocab.txt')
    mention_context_vocab = Vocab('vocab/vocab.txt')
    entity_vocab = Vocab('vocab/entity_vocab.txt')
    print(f'entity_context_vocab size : {len(entity_context_vocab)}')
    print(f'mention_context_vocab size : {len(mention_context_vocab)}')
    print(f'entity_vocab size : {len(entity_vocab)}')


    train_dataset, test_dataset = loading_dataset(entity_context_vocab, mention_context_vocab, entity_vocab)

    model = EntityLink(len(mention_context_vocab),
                       len(entity_context_vocab),
                       len(entity_vocab),
                       emb_dim=128,
                       hid_dim=256)

    model.to('cuda')
    #opt = optimization.BertAdam(model.parameters(), lr=5e-5)
    opt = optim.Adam(model.parameters())
    train(model, train_dataset, test_dataset, opt)