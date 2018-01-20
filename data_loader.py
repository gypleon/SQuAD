import json

'''
input:
output:
'''
def load_data(file_path):
  with open(file_path, 'r') as f:
    data_dict = json.load(f)
  return data_dict

if __name__ == '__main__':
  data_dict = load_data('../data/dev-v1.1.json')
  print(data_dict['version'])
  print(data_dict['data'][0]['title'])
  print(data_dict['data'][0]['paragraphs'][0]['context'])
  print(data_dict['data'][0]['paragraphs'][0]['qas'][0]['id'])
  print(data_dict['data'][0]['paragraphs'][0]['qas'][0]['question'])
  for answer in data_dict['data'][0]['paragraphs'][0]['qas'][0]['answers']:
    print(answer['answer_start'], answer['text'])
