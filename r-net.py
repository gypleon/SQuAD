import tensorflow as tf
import numpy as np
from word_embedding import WORD_EMBEDDING as WE

class R_NET:
  '''
  refer to 
  R-NET: Machine Reading Comprehension with Self-matching Networks
  Natural Language Computing Group, Microsoft Research Asia
  https://www.microsoft.com/en-us/research/publication/mrc/
  '''
  def __init__(self, path_dataset, path_word_embeddings, dimension_word_embeddings = 300,
               length_gru_state = 75, length_attention_state = 75, learning_rate = 1,
               rho = .95, epsilon = 1e-6):
    self.path_ds = path_dataset
    self.dim_we = dimension_word_embeddings
    self.we = WE(path_word_embeddings)
    self.embeddings = self.we.vectors()
    self.len_gs = length_gru_state
    self.len_as = length_attention_state
    self.lr = learning_rate
    self.rho = rho
    self.eps = epsilon
    
  def inference(self, inputs, scope = 'inference'):
    ''' inference model
    input:
      inputs: tuple of passage and question, (passage, question)
    output:
      answer: predicted answer, [start_pointer, end_pointer]
    '''
    passage = inputs[0]
    question = inputs[1]
    # with tf.Graph().as_default():
    with tf.variable_scope(scope):
      # question and passage encoder
      u_Q = self.encode(question)
      u_P = self.encode(passage)

      # gated attention-based rnn
      self.gated_att(u_Q, u_P)

      # self-matching attention

      # output layer
      tf.train.AdadeltaOptimizer(self.lr, self.rho, self.eps)

    return answer

  def encode(self, words):
    ''' encode token sequence as a list of new token representations
    input:
      words: list of tokens
    output:
      representations: list of final representations
    '''
    sequence = [self.embedding(word) for word in words]
    # representations = [self.bidirectionalGRU(sequence[:i+1], 3) for i in range(len(sequence))]
    outputs = self.bidirectionalGRU(sequence, 3)
    representations = tf.squeeze(outputs)
    return representations


  def gated_att(self, seq_1, seq_2):
    '''
    '''
    tf.contrib.rnn.AttentionCellWrapper()
    c = 
    self.GRURNN()
    v = 
    return 

  def self_matching_att(self):
    tf.contrib.rnn.AttentionCellWrapper()

  def input(self):
    return 

  def GRURNN(self, sequence, num_units = self.len_gs, scope = 'GRURNN'):
    '''
    '''
    gru = tf.nn.rnn_cell.GRUCell(num_units)
    return

  def bidirectionalGRU(self, sequence, num_layers, num_units = self.len_gs, scope = 'bidirectionalGRU'):
    ''' bidirectional rnn layer using GRU cells
    input: 
      sequence: sequencial token inputs
      num_layers: number of bidirectionalGRU layers
      num_units: number of dimension of GRU's hidden states
      scope: name of variable scope
    output:
      results: results of biRNN layers, (outputs, output_state_fw, output_state_bw)
        outputs: [batch_size, max_time, layers_output]
        output_state_fw: final states, one tensor per layer, of the forward rnn.
        output_state_bw: similiar to fw above.
    '''
    with tf.variable_scope(scope):
      gru_fw = tf.nn.rnn_cell.GRUCell(num_units)
      gru_bw = tf.nn.rnn_cell.GRUCell(num_units)
      # shape: [batch_size, max_time, ...] -> [1, seq_len, 300+75]
      inputs = np.array([sequence])
      outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([gru_fw] * num_layers, [gru_bw] * num_layers,
      inputs)
    return outputs

  def embedding(self, word, scope = 'embedding'):
    '''
    input:
      word: tokenized word
    output:
      embed: embedding combining both token-level and character-level
    '''
    # word-level embedding
    if word in self.embeddings:
      e = np.array(self.embeddings[word])
    else:
      # OOV
      e = np.array([0] * self.dim_we)
    # char-level embedding
    seq_chars = [self.embeddings[char] for char in word]
    with tf.variable_scope(scope):
      c = self.bidirectionalGRU(seq_chars, 1)
      embed = tf.concat([e, c], axis = 0)
    return embed

if __name__ == '__main__':
