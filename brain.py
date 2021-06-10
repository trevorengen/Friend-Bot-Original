import os
import numpy as np 
import re
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from kmodel import MyModel
from onestep import OneStep
from keras.callbacks import EarlyStopping
import discord
from dotenv import load_dotenv
import sys
import time


with open('michael.txt', 'r', encoding='UTF-8') as file:
	text = file.read()

text = text.replace('\n', ' ')
text_array = text.split(' ')
vocab = sorted(set(text))
print(vocab)
#vocab = sorted(set(text))
print(f'{len(vocab)} unique characters')

chars = tf.strings.unicode_split(text, input_encoding='UTF-8')
ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab), mask_token=None)
ids = ids_from_chars(chars)

chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
					vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

chars = chars_from_ids(ids)

def text_from_ids(ids):
	return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))

ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

seq_length = 25
examples_per_epoch = len(text)//(seq_length+1)

sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(sequence):
	input_text = sequence[:-1]
	target_text = sequence[1:]
	return input_text, target_text

dataset = sequences.map(split_input_target)

BATCH_SIZE = 256
BUFFER_SIZE = 10000

dataset = (
	dataset
	.shuffle(BUFFER_SIZE)
	.batch(BATCH_SIZE, drop_remainder=True)
	.prefetch(tf.data.experimental.AUTOTUNE))

vocab_size = len(vocab)
embedding_dim = 1000
rnn_units = 1024

model = MyModel(
	vocab_size=len(ids_from_chars.get_vocabulary()),
	embedding_dim=embedding_dim,
	rnn_units=rnn_units)

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
val_loss = 4
example_batch_loss = loss(target_example_batch, example_batch_predictions)
mean_loss = example_batch_loss.numpy().mean()

model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_path = r'C:\Users\Trevor\Desktop\AIBot\MyModel'
# Name of the checkpoint files
checkpoint_dir = os.path.dirname(checkpoint_path)

checkpoint_callback = [tf.keras.callbacks.EarlyStopping(monitor='loss',patience=5),
	tf.keras.callbacks.ModelCheckpoint(
    	filepath=checkpoint_path,
    	save_weights_only=True)]

skip_ids = ids_from_chars(['[UNK]'])[:, None]
sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())])
prediction_mask = tf.sparse.to_dense(sparse_mask)

def generate_one_step(inputs, states=None, temperature=.3):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states

def build_model(model):
	print('1. Build a new model.')
	print('2. Continue current models training.')
	print('Pass')
	key = input()
	if(key == '1'):
		print('How many epochs? (Large values will significantly increase training time.)\n')
		EPOCHS = int(input())
		one_step_model = OneStep(model)
		one_step_model.build(input_shape=(None, 1))
		one_step_model.compile(optimizer='adam', loss=loss)
		history = one_step_model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
		print(history)
		one_step_model.save_weights('./training/model_weights')
		return one_step_model
	elif(key == 2):
		pass
	else:
		print('Attempting to load previous model...')
		one_step_model = OneStep(model)
		one_step_model.compile(optimizer='adam', loss=loss)
		latest = tf.train.latest_checkpoint(checkpoint_dir)
		one_step_model.load_weights(latest)
		return one_step_model

one_step_model = build_model(model)
def send_message(start_message, n):
	start = time.time()
	states = None
	next_char = tf.constant([start_message])
	results = [next_char]
	for i in range(n):
		next_char, states = generate_one_step(next_char, states=states)
		results.append(next_char)
	end = time.time()
	result = tf.strings.join(results)
	print('\nRun time:', end - start)
	return result[0].numpy().decode(('utf-8'), '\n\n' + '_'*80)

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

client = discord.Client()

@client.event
async def on_ready():
	print(f'{client.user} has connected to Discord.')

@client.event
async def on_message(message):
	if message.author == client.user:
		return

	# If message content starts with required string will attempt to
	# create a response using the keras OneStep custom model.
	if message.content.startswith('!AMichael'):
		clean_message = ''.join(re.split('!AMichael', message.content))
		print(clean_message)
		message_to_send = send_message(clean_message, 50)
		await message.channel.send(message_to_send)

client.run(TOKEN)