import os
import glob
import pickle as pkl
from main import extract_mention
from pprint import pprint
from multiprocess_generate_data import is_punc

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

def compute_recall(true_mention, predict_mention):
  common = 0
  for t_mention in true_mention:
    for p_mention in predict_mention:
      if t_mention[0] == p_mention[0] and t_mention[1] == p_mention[1]:
        common += 1
        break
      elif p_mention[0] > t_mention[0]:
        break
  if len(true_mention) == 0:
    return int(len(predict_mention) == len(true_mention))
  else:
    return  common / len(true_mention)

def is_bracket_pair(tag):
  dict = {'(':')', '[':']', '《':'》', '【':'】'}
  stack = []
  for i in range(len(tag)):
    if tag[i] == '(' or tag[i] == '[' or tag[i] == '《' or tag[i] == '【':
      stack.append(tag[i])
    elif tag[i] == ')' or tag[i] == ']' or tag[i] == '》' or tag[i] == '】':
      if stack:
        item = stack.pop()
        if item in dict and dict[item] == tag[i]:
          continue
        else:
          return False
      else:
        return False
  return True

def remove_bracket(tag):
  t = ''
  for i in range(len(tag)):
    if tag[i] not in '()[]《》【】':
      t += tag[i]
  return t

def mention_postprocess(mentions):
  new_mentions = []
  for mention in mentions:
    #remove tag no chinese
    if not is_punc(mention[1]):
      text = remove_bracket(mention[1])
      new_mentions.append((mention[0], text))
  return new_mentions

def develop_result_analysis(data_path):
  datas = pkl.load(open(data_path,'rb'))
  print(datas)


#epoch 6
if __name__ == '__main__':
  develop_result_analysis('valid_detect_mention/ensemble.pkl')
  # for data_path in glob.glob('valid_detect_mention/*'):
  #   develop_result_analysis(data_path)
  #is_chinese('?')
  #printS()
  #for file in glob.glob('valid_log/*'):
  #  print(file)
  #  data = pkl.load(open(file,'rb'))
    # for i in range(len(data)):
    #   data[i]['t_tag_mention'] = extract_mention(data[i]['t_tag'], data[i]['tokens'])
    #   data[i]['p_tag_mention'] = extract_mention(data[i]['p_tag'], data[i]['tokens'])
    #   data[i]['p_tag_mention'] = mention_postprocess(data[i]['p_tag_mention'])
    #   data[i]['recall'] = compute_recall(data[i]['t_tag_mention'], data[i]['p_tag_mention'])
    # pkl.dump(data, open(file, 'wb'))
    #print(data)