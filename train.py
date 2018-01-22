import tensorflow as tf
import numpy as np
from data_loader import DATA_LOADER as DL
from r-net import R_NET as rnet

def train():
  with tf.device():
    while loss > 1 or epoch < 25:
      params, loss = rnet.inference()
      update params

if __name__ == '__main__':
  train()
