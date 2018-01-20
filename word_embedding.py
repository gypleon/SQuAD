'''
input: 
  file_path - Glove pre-trained word vector file
output:
  embedding_dict - 300d word structure of word embedding vectors, {word:[300d], ...}
'''
def load_glove(file_path):
  with open(file_path, 'r') as f:
    while True:
      line = f.readline().strip().split(' ')
      if line:
        yield line
      else:
        break

def word_vectors(file_path):
  embedding_dict = dict()
  for line in load_glove(file_path):
    embedding_dict[line[0]] = [float(val) for val in line[1:]]
  return embedding_dict

if __name__ == '__main__':
  embedding = word_vectors('../glove/glove.42B.300d.txt')
  print(embedding['the'])
