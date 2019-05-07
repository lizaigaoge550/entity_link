import json
import os
from pprint import pprint

data_dir = 'ccks2019_el'
datas = [line for line in open(os.path.join(data_dir, 'kb_data'), encoding='utf-8').readlines()]
for data in datas:
    data = json.loads(data)
    if data['subject'] == '比特币':
        pprint(data)