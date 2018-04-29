import tensorflow as tf
import numpy as np
from word_embedding import WORD_EMBEDDING as WE

class R_NET(object):
  '''
  refer to 
  R-NET: Machine Reading Comprehension with Self-matching Networks
  Natural Language Computing Group, Microsoft Research Asia
  https://www.microsoft.com/en-us/research/publication/mrc/
  '''
  def __init__(self, path_dataset, path_vocab, batch_size = 1, train = True, dimension_word_embeddings = 300,
               length_gru_state = 75, length_attention_state = 75, learning_rate = 1, dropout_keep_prob = 0.8,
               rho = .95, epsilon = 1e-6):
    self.path_ds = path_dataset
    self.vocab = WE(path_vocab, dimension_word_embeddings)
    self.batch_size = batch_size
    self.len_gs = length_gru_state
    self.len_as = length_attention_state
    self.lr = learning_rate
    self.keep_prob = dropout_keep_prob
    self.rho = rho
    self.eps = epsilon
    self.graph = self.build()
    
  @property
  def answer(self):
    return self.answer

  def build(self, scope = 'rnet'):
    ''' build model graph
    input:
    output:
    '''
    with tf.variable_scope(scope):
      # placeholders
      questions = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length, ])
      passages = tf.placeholder()

      # question and passage encoder
      u_Q = self.encode(questions)
      u_P = self.encode(passages)

      # gated attention-based rnn
      gated_u_P = self.gated_attn(u_Q, u_P)

      # self-matching attention
      h_P = self.self_matching_attn(gated_u_P)

      # output layer - pointer networks
      self.answer = output_layer(h_P, u_Q)
      self.losses = tf.losses.softmax_cross_entropy(truths, self.answer)
    return 

  def inference(self):
    ''' inference
    input:
      inputs: tuples of passage, question and answer: (passage, question, truth)
    output:
      answer: predicted answers, [start_pointer, end_pointer]
      losses: 
    '''
    passages = tf.slice(inputs, [], [])
    questions = tf.slice(inputs, [], [])
    truths = tf.slice(inputs, [], [])
    return self.answer, self.losses

  def train(self):
    optimizer = tf.train.AdadeltaOptimizer(learning_rate = self.lr, beta1 = self.rho, epsilon = self.eps)
    optimizer.minimize(self.losses)

  def encode(self, words):
    ''' encode token sequence as a list of new token representations
    input:
      words: list of tokens
    output:
      representations: list of final representations
    '''
    # TODO: new biRNN for each word? how to batch words for embedding?
    # build graph for the longest word, traverse corpus in advance?
    sequence = [self.embedding(word) for word in words]
    outputs = self.bidirectionalGRU(sequence, 3)
    representations = tf.squeeze(outputs)
    return representations

  def gated_attn(self, seq_1, seq_2, num_units = self.len_gs, scope = 'GatedAttn'):
    ''' gated attention for seq_2 w.r.t seq_1
      input:
        seq_1: query sequence in attention mechanism
        seq_2: encoder sequence in attention mechanism
      output:
        outputs:
        v_P:
        context:
    '''
    u_Q = seq_1
    u_P = seq_2
    cell = self.GRUCellGPU(num_units)
    with tf.variable_scope(scope):
      attn_v = tf.get_variable('attn_v', [num_units])
      w_u_Q = tf.get_variable('weight_ques')
      w_u_P = tf.get_variable('weight_pass_orig')
      w_v_P = tf.get_variable('weight_pass_ques')
      v_P = tf.get_variable('embedding_pass_ques')
      score = tf.reduce_sum(attn_v * tf.tanh(w_u_Q * u_Q + w_u_P * u_P + w_v_P * v_P), [2]) # scores of all tokens
      alignments = tf.softmax(score)
      alignments = tf.expand_dims(alignments, 1)
      u_Q = tf.expand_dims(u_Q, 1)
      context = tf.reduce_sum(tf.matmul(alignments, u_Q, transpose_b=True), [2])
      # gate
      inputs = tf.concat([u_Q, context], 1)
      w_g = tf.get_variable('weight_gate')
      g = tf.sigmoid(tf.reduce_sum(w_g * inputs))
      gated_inputs = g * inputs
      outputs, v_P = cell(gated_inputs, v_P)
    return outputs, v_P

  def self_matching_attn(self, seq, scope = 'SelfMatchAttn'):
    ''' self-matching attention of seq
      input:
      output:
    '''
    v_P = seq
    with tf.variable_scope(scope):
      attn_v = tf.get_variable('attn_v')
      w_v_P = tf.get_variable('weight_passage')
      w_v_P_w = tf.get_variable('weight_passage_wave')
      score = tf.reduce_sum(attn_v * tf.tanh(w_v_P * v_P + w_v_P_w * v_P), [2]) # scores for all tokens
      alignments = tf.softmax(score)
      alignments = tf.expand_dims(alignments, 1)
      v_P = tf.expand_dims(u_Q, 1)
      context = tf.reduce_sum(tf.matmul(alignments, v_P, transpose_b=True), [2])
      inputs = tf.concat([seq, context], 1)
      outputs = self.bidirectionalGRU(inputs, 1)
      h_P = outputs
    return h_P

  def output_layer(self, seq_1, seq_2, scope = 'PointerNetwork'):
    h_P = seq_1
    u_Q = seq_2
    cell = self.GRUCellGPU(num_units)
    with tf.variable_scope(scope):
      # initialize hidden state of answer
      attn_v_a = tf.get_varialbe('attn_v')
      w_u_Q = tf.get_varialbe('weight_passage')
      w_v_Q = tf.get_varialbe('weight_answer')
      V_Q = tf.get_varialbe('weight_answer')
      score_a = tf.reduce_sum(attn_v_a * tf.tanh(w_u_Q * u_Q + w_a * h_a), [2]) # scores for all tokens
      alignments_a = tf.softmax(score_a)
      r_Q = tf.reduce_sum(tf.matmul(alignments_a, u_Q, transpose_b=True), [2])

      attn_v = tf.get_varialbe('attn_v')
      w_P = tf.get_varialbe('weight_passage')
      w_a = tf.get_varialbe('weight_answer')
      h_a = tf.get_varialbe('embedding_answer')
      score = tf.reduce_sum(attn_v * tf.tanh(w_P * h_P + w_a * h_a), [2]) # scores for all tokens
      alignments = tf.softmax(score)
      alignments = tf.expand_dims(alignments, 1)
      v_P = tf.expand_dims(u_Q, 1)
      context = tf.reduce_sum(tf.matmul(alignments, v_P, transpose_b=True), [2])
      outputs, h_a = cell(h_a, context)
      self.answer = outputs
    return self.answer

  def GRUCellGPU(self, num_units = self.len_gs, gpu_num = 0, scope = 'GRUCellGPU'):
    ''' wrapper GRUCell with Dropout & GPU device
    '''
    gru_cell = tf.contrib.rnn.DeviceWrapper(
      tf.contrib.rnn.DropoutWrapper(
        tf.nn.rnn_cell.GRUCell(num_units),
        input_keep_prob = self.keep_prob, output_keep_prob = self.keep_prob, state_keep_prob = self.keep_prob),
      "/device:GPU:%i" % gpu_num)
    return gru_cell

  def bidirectionalGRU(self, sequence, num_layers, num_units = self.len_gs, scope = 'bidirectionalGRU'):
    ''' bidirectional rnn layer using GRU cells
    input: 
      sequence: sequencial token/char inputs
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
      gru_fw = self.GRUCellGPU(num_units)
      gru_bw = self.GRUCellGPU(num_units)
      # shape: [batch_size, max_time_step, ...]
      # TODO: decide batch_size_word based on the batch with the largest amount of unique words. 
      # char embedding: [batch_size_word, word_length, 300], 
      # sentence: [batch_size, seq_len, 300+75]
      inputs = np.array([sequence])
      outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([gru_fw] * num_layers, [gru_bw] * num_layers,
      inputs)
    return outputs

  def embedding(self, word, scope = 'embedding'):
    '''
    input:
      word: tokenized word
    output:
      embed: embedding combining both word-level and character-level
    '''
    word_embed, char_embeds = self.vocab.embedding(word)
    with tf.variable_scope(scope):
      # char_embeds: [word_length, vocab_dim]
      # TODO: dynamic word_length needs padding?
      # based on the longest word, traverse corpus in advance.
      char_embed = self.bidirectionalGRU(char_embeds, 2)
      embed = tf.concat([word_embed, char_embed], axis = 0)
    return embed
    

if __name__ == '__main__':
