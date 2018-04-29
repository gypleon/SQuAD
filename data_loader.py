import json
import numpy as np
from pycorenlp import StanfordCoreNLP

class DATA_LOADER(object):
  def __init__(self, data_path):
    '''
    intput:
      data_path: path of training/development data set
    output:
    '''
    self.corenlp_url = 'http://localhost:9000'
    self.nlp = StanfordCoreNLP(self.corenlp_url)
    self.data_path = data_path

  def load_data(self, is_test = False, is_generator = True):
    '''
    input:
    output:
      doc_list: list of tokenized documents
      data_dict: dict of original documents
    '''
    with open(self.data_path, 'r') as f:
      data_dict = json.load(f)
    # preprocessing
    doc_list = []
    for doc in data_dict['data']:
      # doc_list.append(doc['title'])
      para_list = []
      for para in doc['paragraphs']:
        para_dict = {}
        qas_list = []
        para_dict['context'] = self.tokenize(para['context'])
        for qa in para['qas']:
          qa_dict = {'question': self.tokenize(qa['question'])}
          a_list = []
          for a in qa['answers']:
            a_list.append({'text': self.tokenize(a['text']), 'answer_start': a['answer_start']})
          qa_dict['answers'] = a_list
          qas_list.append(qa_dict)
        para_dict['qas'] = qas_list
        if is_generator:
          yield para_dict
        else:
          para_list.append(para_dict)
      if not is_generator:
        doc_list.append(para_list)
    if not is_generator:
      if not is_test:
        # release memory
        data_dict = None
        return doc_list
      else:
        return doc_list, data_dict

  def batch(self):
    # TODO: generate a batch of passages? but how to deal with dynamic amounts of question-answer pairs?
    # 
    return batch

  def tokenize(self, text):
    '''
    input:
      text: original text string
    output:
      _: list of tokens
    '''
    tokens = self.nlp.annotate(text, properties={'annotators': 'tokenize','outputFormat': 'json'})
    return [token['word'] for token in tokens['tokens']]
  
if __name__ == '__main__':
  data_loader = DATA_LOADER('../data/dev-v1.1.json')
  for para in data_loader.load_data():
    print(len(para['context']), len(para['qas']))

  ''' test in details
  doc_list, data_dict = data_loader.load_data(is_test = True, is_generator = False)
  print(data_dict['version'])
  print(data_dict['data'][0]['title'])
  print(data_dict['data'][0]['paragraphs'][0]['context'])
  print(data_dict['data'][0]['paragraphs'][0]['qas'][0]['id'])
  print(data_dict['data'][0]['paragraphs'][0]['qas'][0]['question'])
  for answer in data_dict['data'][0]['paragraphs'][0]['qas'][0]['answers']:
    print(answer['answer_start'], answer['text'])

  print(doc_list[0][0])
  '''
