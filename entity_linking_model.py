import torch
import torch.nn as nn
from basemodel import BaseModel
from torch.nn import init
import torch.nn.functional as F

class EntityLink(BaseModel):
    def __init__(self, mention_vocab_size, entity_vocab_size, emb_dim, hid_dim):
        super(EntityLink, self).__init__(mention_vocab_size, emb_dim, hid_dim)
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.mention_vocab_size = mention_vocab_size
        self.entity_vocab_size = entity_vocab_size

        self.mention_embedding = nn.Embedding(mention_vocab_size, emb_dim)
        self.entity_embedding = nn.Embedding(entity_vocab_size, emb_dim)

        self.attention_w = nn.Linear(2*hid_dim, 1)

        init.normal_(self.mention_embedding.weight, -0.01, 0.01)
        init.normal_(self.entity_embedding.weight, -0.01, 0.01)
        init.xavier_normal_(self.attention_w)

    def soft_attention(self, embs, poss):
        res = []
        for (emb, pos) in zip(embs.chunk(dim=0), poss.chunk(dim=0)):
            emb = emb[poss[0]:poss[1]+1, :]
            attention = self.attention_w(emb)
            attention = F.softmax(attention, dim=0)
            res.append(torch.sum(attention * emb, dim=0))
        return torch.stack(res, dim=0) #batch, hid_dim


    def generate_representation(self, context_emb, position):
        '''
        :param context_emb: batch, seq_len, emb_dim
        :param position: batch, 2
        :return:
        '''
        assert context_emb.size(0) == position.size(0)
        position = position.unsqueeze(-1).repeat(1,1,context_emb.size(-1))
        emb = torch.gather(context_emb, 1, position) #batch, 2, hid_dim
        soft_attention_emb = self.soft_attention(context_emb, position)
        soft_attention_emb = soft_attention_emb.unsqueeze(1)
        res = torch.cat((emb, soft_attention_emb), dim=1) #batch, 3, hid_dim
        res = res.view(context_emb.size(0), -1) #batch, 3*hid_dim
        return res.contiguous()

    def forward(self, mention_contexts, mention_position, entity_contexts, entity_position, target):
        '''
        mention_contexts : batch, m_seq_len
        mention_position: batch, 2 (start, end)
        entity_contexts : batch, candidates, e_seq_len
        entity_position : batch, candidates, 2 (start, end)
        target : batch
        '''
        #first embedding_layer
        mention_seq_len = torch.sum(mention_contexts > 0, dim=-1)
        mention_contexts_emb = self.mention_embedding(mention_contexts)

        mention_contexts_emb, _ = self._lstm_forward(mention_contexts_emb, mention_seq_len)
        mention_emb = self.generate_representation(mention_contexts_emb, mention_position)

        entity_embs = []
        for entity_context in entity_contexts.chunk(dim=1):
            entity_seq_len = torch.sum(entity_context > 0, dim=1)
            entity_contexts_emb = self.entity_embedding(entity_context)
            entity_contexts_emb, _ = self._lstm_forward(entity_contexts_emb, entity_seq_len)
            entity_emb = self.generate_representation(entity_contexts_emb, entity_position)
            entity_embs.append(entity_emb)
        entity_embs = torch.stack(entity_embs, dim=1) #batch, cands, 3*hid_dim

        scores = torch.sum(mention_emb.unsqueeze(1) * entity_embs, dim=-1) #batch, cands
        scores = F.softmax(scores, dim=-1)

        loss = F.multi_margin_loss(scores, target, margin=0.5)

        return scores, loss


