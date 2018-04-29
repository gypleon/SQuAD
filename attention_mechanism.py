import tensorflow as tf
from tensorflow.contrib.seq2seq import AttentionWrapper

class GatedAttention(tf.contrib.seq2seq.BahdanauAttention):
  ''' implementation of gated attention-based recurrent networks
  '''
  def __call__(self, u_Q, u_P, scope = 'GatedAttn'):
    with tf.variable_scope(scope):
      attn_v = tf.get_variable('attn_v', [num_units])
      w_u_Q = tf.get_variable('weight_ques')
      w_u_P = tf.get_variable('weight_pass_orig')
      w_v_P = tf.get_variable('weight_pass_ques')
      v_P = tf.get_variable('embedding_pass_ques')
      score = tf.reduce_sum(tf.attn_v * tf.tanh(w_u_Q * u_Q + w_u_P * u_P + w_v_P * v_P), [2])
    alignments = self._probability_fn(score)
    return alignments

class SelfMatchingAttention(tf.contrib.seq2seq.BahdanauAttention):
  ''' implementation of self-matching attention
  '''
  def _init_(self):

  def score(self):

