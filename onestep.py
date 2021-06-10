import os
import numpy as np 
import re
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from kmodel import MyModel

class OneStep(tf.keras.Model):
  def __init__(self, model, temperature=.4):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.embedding = model.embedding
    self.gru = model.gru
    self.dense = model.dense

    # Create a mask to prevent "[UNK]" from being generated.
    # skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
    # sparse_mask = tf.SparseTensor(
    #     # Put a -inf at each bad index.
    #     values=[-float('inf')]*len(skip_ids),
    #     indices=skip_ids,
    #     # Match the shape to the vocabulary
    #     dense_shape=[len(ids_from_chars.get_vocabulary())])
    # self.prediction_mask = tf.sparse.to_dense(sparse_mask)

 # @tf.function
  # def generate_one_step(self, inputs, states=None):
  #   # Convert strings to token IDs.
  #   input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
  #   input_ids = self.ids_from_chars(input_chars).to_tensor()

  #   # Run the model.
  #   # predicted_logits.shape is [batch, char, next_char_logits]
  #   predicted_logits, states = self.model(inputs=input_ids, states=states,
  #                                         return_state=True)
  #   # Only use the last prediction.
  #   predicted_logits = predicted_logits[:, -1, :]
  #   predicted_logits = predicted_logits/self.temperature
  #   # Apply the prediction mask: prevent "[UNK]" from being generated.
  #   predicted_logits = predicted_logits + self.prediction_mask

  #   # Sample the output logits to generate token IDs.
  #   predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
  #   predicted_ids = tf.squeeze(predicted_ids, axis=-1)

  #   # Convert from token ids to characters
  #   predicted_chars = self.chars_from_ids(predicted_ids)

  #   # Return the characters and model state.
  #   return predicted_chars, states
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