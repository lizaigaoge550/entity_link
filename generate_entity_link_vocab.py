import pickle as pkl
from collections import defaultdict, Counter

def generate_vocab():
    entity_dict = set()
    text_dict = defaultdict(int)
    kb_data = pkl.load(open('kb_info/kb_dict.pkl', 'rb'))
    for key, value in kb_data.items():
        text = value['text']
        subject = key
        if subject not in entity_dict:
            entity_dict.add(subject)
        else:
            raise Exception(f'entity : {subject} duplicated!!!')
        for t in text:
            text_dict[t] += 1
    entity_vocab = open('vocab/entity_vocab.txt','w',encoding='utf-8')
    entity_context_vocab =open('vocab/entity_context_vocab.txt', 'w', encoding='utf-8')
    for entity in entity_dict:
        entity_vocab.write(entity+'\n')
    entity_context_vocab.write('[PAD]\n')
    entity_context_vocab.write('[UNK]\n')
    for token, value in Counter(text_dict).most_common():
        entity_context_vocab.write(token+'\n')


if __name__ == '__main__':
    generate_vocab()