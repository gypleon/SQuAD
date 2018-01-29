class WORD_EMBEDDING:
  '''
  input:
    file_path: path of pre-trained word embeddings library
  output:
  '''
  def __init__(self, file_path):
    self.file_path = file_path
    return

  def load_glove(self):
    '''
    input: 
    output:
      embedding_dict - 300d word representation of word embedding vectors, {word:[300d], ...}
    '''
    with open(self.file_path, 'r') as f:
      while True:
        line = f.readline()
        if line:
          line = line.strip().split(' ')
          yield line
        else:
          break

  def vectors(self):
    '''
    input:
    output:
      embedding_dict: dict of embedding vectors
    '''
    embedding_dict = {}
    for line in self.load_glove():
      embedding_dict[line[0]] = [float(val) for val in line[1:]]
    return embedding_dict

if __name__ == '__main__':
  word_embedding = WORD_EMBEDDING('../glove/glove.840B.300d.txt.test')
  embeddings = word_embedding.vectors()
  print(embeddings['the'])
  print(len(embeddings['the']))
