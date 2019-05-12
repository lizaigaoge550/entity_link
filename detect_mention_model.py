from basemodel import BaseModel
from allennlp.modules import ConditionalRandomField
import torch
import torch.nn as nn
import torch.nn.init as init
from pytorch_pretrained_bert import modeling
'''
this model is lstm-crf model
'''
class DetectModel(BaseModel):
    def __init__(self,  vocab_size, input_dim, output_dim, num_tags,
                 constrains,
                 include_start_end_transitions=False,n_layer=1):
        super(DetectModel, self).__init__(vocab_size=vocab_size, input_dim=input_dim, n_layers=n_layer,
                                          output_dim=output_dim)
        self.numtags = num_tags
        if self.bert_weight:
            self.tag_project_layer = nn.Linear(2*output_dim, num_tags)
        else:
            self.tag_project_layer = nn.Linear(2*output_dim, num_tags)
        self.crf = ConditionalRandomField(num_tags, constrains, include_start_end_transitions)
        init.xavier_normal_(self.tag_project_layer.weight)

    def forward(self, tokens, tags):
        '''
        :param tokens: batch, seq_len
        :param tags: batch, seq_len
        :return:
        '''
        #first embeding layer
        mask = tokens > 0
        token_lens = torch.sum(mask, dim=-1).long()
        assert len(token_lens) == tokens.size(0)

        token_emb = self.embedding(tokens)
        encoder_output, _ = self._lstm_forward(token_emb, token_lens) #[encoder_output, max_seq_len, hid_dim]


        #now trunk tags to max_seq_len
        tags = tags[:,:encoder_output.size(1)]

        #crf,
        logits = self.tag_project_layer(encoder_output)
        mask = mask[:, :logits.size(1)]
        best_paths = self.crf.viterbi_tags(logits, mask)
        #inputs: torch.Tensor, tags: torch.Tensor, mask
        loss = self.crf(logits, tags, mask)
        return loss, best_paths, token_lens

    def bert_forward(self, tokens, offset, tags, is_training=True):
        mask = offset > 0
        token_lens = torch.sum(mask, dim=-1).long()
        encoder_output = self.use_bert(tokens, offset)
        encoder_output, _ = self._lstm_forward(encoder_output, token_lens)
        logits = self.tag_project_layer(encoder_output)
        if not is_training:
            best_paths = self.crf.viterbi_tags(logits, mask)
        # inputs: torch.Tensor, tags: torch.Tensor, mask
        loss = self.crf(logits, tags, mask)
        if is_training:
            return loss
        else:
            return loss, best_paths