import tensorflow as tf
import numpy as np
from word_embedding import WORD_EMBEDDING as WE

class R_NET:
  def __init__(self, path_dataset, path_word_embeddings,
               learning_rate = 1, rho = .95, epsilon = 1e-6):
    self.path_ds = path_dataset
    self.path_we = path_word_embeddings
    self.lr = learning_rate
    self.rho = rho
    self.eps = epsilon
    
  def inference(self):
    # question and passage encoder
    for word in passage:
      u = self.embedding(word)
    tf.contrib.rnn.stack_bidirectional_rnn()
    tf.contrib.rnn.stack_bidirectional_rnn()
    tf.contrib.rnn.stack_bidirectional_rnn()

    # gated attention-based rnn
    tf.contrib.rnn.stack_bidirectional_rnn()
    tf.contrib.rnn.AttentionCellWrapper()

    # self-matching attention
    tf.contrib.rnn.AttentionCellWrapper()

    # output layer
    tf.train.AdadeltaOptimizer(self.lr, self.rho, self.eps)

    return

  def embedding(self, word):
    we = WE(path_we)
    # word-level embedding
    if word in we:
      e = np.ndarray(we[word])
    else:
      # OOV
      e = np.ndarray([0 for i in range(300)]
    # char-level embedding
    tf.nn.rnn_cell.GRUCell()
    c = tf.contrib.rnn.stack_bidirectional_rnn()
    u = tf.contrib.rnn.stack_bidirectional_rnn()
    return u

if __name__ == '__main__':
