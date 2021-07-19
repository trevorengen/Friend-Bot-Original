
import os
import numpy as np 
import re
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing

class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    # Input layer: Is a trainable lookup table that maps each character-ID
    # to a vector with embedding_dim dimensions. Would like to know more
    # regarding how the dimensions effect the probabilities of the dense layer.
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    # A type of recurrent neural net with RNN units = rnn_units.
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)

    # Output layer: selecting what character to use next by comparing the log
    # likelihood of the next character according to the model.
    self.dense = tf.keras.layers.Dense(vocab_size)
  @tf.function
  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x