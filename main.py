import tensorflow as tf 
from tensorflow import keras
import time
from onestep import OneStep
from kmodel import MyModel
from brain import chars_from_ids, ids_from_chars, vocab_size, vocab, BATCH_SIZE, BUFFER_SIZE, dataset
import os
import tempfile


model = keras.models.load_model(r'C:\Users\Trevor\Desktop\AIBot\this_model')
one_step = OneStep(model, chars_from_ids, ids_from_chars)
one_step.build(input_shape=(None, 1))
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss)
def generate_text():

	states = None
	next_char = tf.constant(['Hey you fucko '])
	results = [next_char]
	for n in range(100):
		next_char, states = one_step.generate_one_step(next_char, states=states)
		results.append(next_char)
		print(results)
	print(tf.strings.join(result[0].numpy().decode('utf-8')))

generate_text()