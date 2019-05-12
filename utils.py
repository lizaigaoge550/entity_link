import torch
import numpy as np
import torch.nn as nn

def get_final_encoder_states(encoder_outputs, mask, bidirectional=False):
    last_word_indices = mask.sum(1).long - 1
    batch_size, _, encoder_output_dim = encoder_outputs.size()
    expanded_indices = last_word_indices.view(-1, 1, 1).expand(batch_size, 1, encoder_output_dim)
    final_encoder_output = encoder_outputs.gather(1, expanded_indices)
    final_encoder_output = final_encoder_output.squeeze(1)
    if bidirectional:
        final_forward_output = final_encoder_output[:, :(encoder_output_dim // 2)]
        final_backward_output = encoder_outputs[:, 0, (encoder_output_dim // 2)]
        final_encoder_output = torch.cat([final_forward_output, final_backward_output], dim=-1)
    return final_encoder_output


def get_mask(lens):
    '''
    :param lens: list of batch, every item is a int means the length of a sample
    :return: [batch, max_seq_len]
    '''
    max_len = max(lens)
    batch_size = len(lens)
    seq_range = torch.arange(max_len).long().to('cuda')
    seq_range = seq_range.unsqueeze(0).expand(batch_size, max_len)

    seq_length = lens.unsqueeze(1).expand(batch_size, max_len)
    mask = seq_range < seq_length
    return mask.float()


def get_mask_2(lens):
    max_len = max(lens)
    batch_size = len(lens)
    mask = torch.FloatTensor(batch_size, max_len)
    mask.fill_(0)
    for i, l in enumerate(lens):
        mask[i, :l]._fill(1.0)
    return mask



def get_char_mask(lens):
    '''
    :param lens: list of batch, in particularly, every item is a list, and every item in the list means the length of word
    :return: mask [batch, max_seq_len, max_word_len]
    '''
    max_seq_len = torch.max(torch.sum(lens > 0, dim=-1)).long().item()
    tensor_len = torch.zeros((lens.size(0), max_seq_len))

    #first trunk every len to max_seq_len
    for i in range(lens.size(0)):
            tensor_len[i] = lens[i,:max_seq_len]

    batch_size, seq_len = tensor_len.size()
    max_word_len = torch.max(tensor_len).int().item()
    seq_range = torch.arange(max_word_len).long()
    seq_range = seq_range.view(1,1,max_word_len).expand(batch_size, seq_len, max_word_len)

    seq_length = lens.unsqueeze(-1).expand(batch_size, seq_len, max_word_len)
    mask = seq_range < seq_length
    return mask.float()



def lstm_encoder(sequence, lstm, seq_lens, init_states, is_mask=False, get_final_output=False):
    batch_size = sequence.size(0)
    if isinstance(seq_lens, torch.Tensor):
        seq_lens_value = seq_lens.tolist()
    else:
        seq_lens_value = seq_lens
    assert len(seq_lens_value) == batch_size
    sort_ind = np.argsort(seq_lens_value)[::-1].tolist()
    sort_seq_lens = [seq_lens_value[i] for i in sort_ind]
    emb_sequence = reorder_sequence(sequence, sort_ind)

    init_states = (init_states[0].contiguous(), init_states[1].contiguous())

    packed_seq = nn.utils.rnn.pack_padded_sequence(emb_sequence, sort_seq_lens, batch_first=True)
    packed_out, final_states = lstm(packed_seq, init_states)
    lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
    back_map = {ind : i for i, ind in enumerate(sort_ind)}
    reorder_ind = [back_map[i] for i in range(len(sort_ind))]
    lstm_out = reorder_sequence(lstm_out, reorder_ind)
    final_states = reorder_lstm_states(final_states, reorder_ind)

    if is_mask:
        mask = get_mask(seq_lens) # batch, max_seq_lens
        assert lstm_out.size(1) == mask.size(1)
        lstm_out *= mask.unsqueeze(-1)
        return lstm_out, final_states #[batch, max_seq_lens, hid_dim], ([n_layer, batch, hid_dim], [n_layer, batch, hid_dim])

    if get_final_output:
        mask = get_mask(seq_lens)
        lstm_out = get_final_encoder_states(lstm_out, mask, bidirectional=True)
        return lstm_out, final_states #[batch, hid_dim], ([n_layer, batch, hid_dim], [n_layer, batch, hid_dim])


def reorder_sequence(emb_sequence, order):
    order = torch.LongTensor(order).to('cuda')
    return emb_sequence.index_select(index=order, dim=0)



def reorder_lstm_states(states, order):
    assert isinstance(states, tuple)
    assert len(states) == 2
    assert states[0].size() == states[1].size()
    assert len(order) == states[0].size()[1]

    order = torch.LongTensor(order).to('cuda')
    sorted_states = (states[0].index_select(index=order, dim=1), states[1].index_select(index=order, dim=1))
    return sorted_states


def tokens_to_indices(tokens, vocab, tokenizer, max_pieces=512):
    wordpiece_ids = [vocab.word2id('[CLS]')]
    offset = 1
    offsets = []
    for token in tokens:
        text = token.lower()
        token_wordpiece_ids = [vocab.word2id(wordpiece) for wordpiece in tokenizer.tokenize(text)]
        if len(wordpiece_ids) + len(token_wordpiece_ids) + 1 <= max_pieces:
            offsets.append(offset)
            offset += len(token_wordpiece_ids)
            wordpiece_ids.extend(token_wordpiece_ids)
        else:
            break
    wordpiece_ids.extend([vocab.word2id('[SEP]')])
    return wordpiece_ids, offsets



def save_to_max_nozero_len(input):
    batch_size, seq_len = input.size()
    max_len = torch.max(torch.sum(input > 0, dim=-1)).long().item()
    for i in range(len(batch_size)):
        input[i] = input[i, :max_len]
    return input, max_len


def collate_fn(batches):
    token_max_len = 0
    offset_max_len = 0
    for batch in batches:
        token_max_len = max(token_max_len, len(batch['tokens']))
        offset_max_len = max(offset_max_len, len(batch['offset']))
    #padding
    for batch in batches:
        batch['tokens'] = batch['tokens'] + [0] * (token_max_len - len(batch['tokens']))
        batch['offset'] = batch['offset'] + [0] * (offset_max_len - len(batch['offset']))
        batch['tags'] = batch['tags'] + [0] * (offset_max_len - len(batch['tags']))


    tokens = torch.stack([torch.LongTensor(batch['tokens']) for batch in batches])
    tags = torch.stack([torch.LongTensor(batch['tags']) for batch in batches])
    offset = torch.stack([torch.LongTensor(batch['offset']) for batch in batches])
    original_tokens = [batch['origin_tokens'] for batch in batches]
    text_ids = [batch['text_id'] for batch in batches]
    return {'tokens':tokens,  'tags' : tags, 'offset':offset,
            'original_tokens':original_tokens, 'text_ids':text_ids}

def collate_fn_entity_link(batches):
    '''
    'mention_context, '
    'mention_position, '
    'entity_cands_id, '
    'entity_contexts_id'
    '''
    mention_context_max_len = 0
    entity_contexts_max_len = 0
    for batch in batches:
        mention_context_max_len = max(len(batch['mention_context']), mention_context_max_len)
        entity_contexts_max_len = max(len(max(batch['entity_contexts_id'], len)), entity_contexts_max_len)
    mention_context = []
    entity_context = []
    pos = []
    cand_id = []
    for batch in batches:
        mention_context.append(torch.LongTensor(
            batch['mention_context'] + [0]*(mention_context_max_len - len(batch['mention_context']))))

        p = [context + [0]*(entity_contexts_max_len - len(context)) for context in batch['entity_contexts_id']]
        entity_context.append(torch.LongTensor(p))
        pos.append(batch['mention_position'])
        cand_id.append(torch.LongTensor(batch['entity_cands_id']))
    return {'mention_context':torch.stack(mention_context, dim=0),
            'mention_position':pos,
            'entity_cands_id':torch.stack(cand_id, dim=0),
            'entity_contexts_id':torch.stack(entity_context, dim=0)
            }












