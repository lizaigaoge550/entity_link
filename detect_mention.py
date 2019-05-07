import json

def get_subject_dict():
    subject_dict = {}
    kb_datas = [line for line in open('ccks2019_el\\kb_data', encoding='utf-8').readlines()]
    for kb_data in kb_datas:
        kb_data = json.loads(kb_data)
        subject = kb_data['subject']
        subject_id = kb_data['subject_id']
        if subject_id not in subject_dict:
            subject_dict[subject_id] = subject
        else:
            raise Exception('Entity : {} in subject_dict'.format(subject))
    return subject_dict

def get_data_mention_info():
    data_mention_info = {}
    datas = [line for line in open('ccks2019_el\\train.json', encoding='utf-8').readlines()]
    for data in datas:
        data = json.loads(data)
        text_id = data['text_id']
        mention_info = data['mention_data']
        data_mention_info[text_id] = mention_info
    return data_mention_info



def detect_mention():
    count  = 0
    total = 0
    no_exist = 0
    subject_dict = get_subject_dict()
    data_mention_info = get_data_mention_info()
    for key, valuelist in data_mention_info.items():
        for value in valuelist:
            kb_id = value['kb_id']
            mention = value['mention']
            if kb_id in subject_dict:
                if subject_dict[kb_id] != mention:
                    print('key : {} subject_mention : {}, data_mention :{}'.format(key, subject_dict[kb_id], mention))
                    count += 1
            else:
                print('kb_id : {} not exist'.format(kb_id))
                no_exist += 1
            total += 1
    print(no_exist)
    print(count)
    print(total)


if __name__ == '__main__':
    detect_mention()
