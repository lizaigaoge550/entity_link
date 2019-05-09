import os
import glob
import pickle as pkl
from main import extract_mention
from pprint import pprint


def printS():
  count = 0
  datas = pkl.load(open('data.pkl','rb'))
  for data in datas:
    if 'S' in data['tags'] and count < 10:
      pprint(data)
      count += 1


def is_chinese(c):
  if  '\u4e00' <= c <= '\u9fff':
    print(True)

if __name__ == '__main__':
  is_chinese('?')
  #printS()
  #for file in glob.glob('valid_log/*'):
    #print(file)
    #data = pkl.load(open(file,'rb'))
    #for i in range(len(data)):
    #  data[i]['t_tag_mention'] = extract_mention(data[i]['t_tag'], data[i]['tokens'])
    #  data[i]['p_tag_mention'] = extract_mention(data[i]['p_tag'], data[i]['tokens'])
    #pkl.dump(data, open(file, 'wb'))
    #print(data)