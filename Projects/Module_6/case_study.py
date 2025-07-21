restart = True
epoch_to_pickup = 0

from tensorflow.keras.layers import StringLookup
import numpy as np
import os
import time
import random
import contextlib
import io
import re
import string
import gc
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
path = r"C:\Users\ajvas\Documents\GitHub\CSE450\Projects\Module_6"



### Process text ###

def preprocess_text(text):

    text = text.replace("Project Gutenberg", "")
    text = text.replace("Gutenberg", "")

    # Remove carriage returns
    text = text.replace("\r", "")

    # fix quotes
    text = text.replace("“", "\"")
    text = text.replace("”", "\"")

    # Replace any capital letter at the start of a word with ^ followed by the lowercase letter
    text = re.sub(r"(?<![a-zA-Z])([A-Z])", lambda match: f"^{match.group(0).lower()}", text)

    # Replace all other capital letters with lowercase
    text = re.sub(r"([A-Z])", lambda match: f"{match.group(0).lower()}", text)

    # Remove duplicate whitespace
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\t+", "\t", text)

    # Replace whitespace characters with special words
    text = re.sub(r"(\t)", r" zztabzz ", text)
    text = re.sub(r"(\n)", r" zznewlinezz ", text)
    text = re.sub(r"(\s)", r" zzspacezz ", text)

    # Split before and after punctuation
    for punctuation in string.punctuation:
        text = text.replace(punctuation, f" {punctuation} ")

    return text

def postprocess_text(text):
    text = text.replace("zztabzz", "\t")
    text = text.replace("zznewlinezz", "\n")
    text = text.replace("zzspacezz", " ")

    text = re.sub(r"\^([a-z])", lambda match: f"{match.group(1).upper()}", text)

    text = text.replace("^", "")

    return text

def getMyText():
    file_name = 'mark_twain2.txt'
    file_url = 'https://www.gutenberg.org/cache/epub/3186/pg3186.txt'
    local_dir = r"C:\Users\ajvas\Documents\GitHub\CSE450\Projects\Module_6"
    local_path = os.path.join(local_dir, file_name)

    try:
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        if os.path.exists(local_path):
            print(f"File '{file_name}' found locally. Using it.")
        else:
            print(f"File '{file_name}' not found locally. Downloading it.")
            downloaded_path = tf.keras.utils.get_file(file_name, file_url)

            with open(downloaded_path, 'rb') as source_file:
                with open(local_path, 'wb') as dest_file:
                    dest_file.write(source_file.read())

        with open(local_path, 'rb') as file:
            text = file.read().decode(encoding='utf-8')

        return preprocess_text(text)

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def getRandomText(numbooks = 1, verbose=False):
  download_log = io.StringIO()
  text_random = ''
  for b in range(numbooks):
    foundbook = False
    while(foundbook == False):
      booknum = random.randint(100,60000)
      if verbose:
        print('Trying Book #: ',booknum)
      if random.random() > 0.5:
        url = 'https://www.gutenberg.org/files/' + str(booknum) + '/' + str(booknum) + '-0.txt'
        filename_temp = str(booknum) + '-0.txt'
      else:
        url = 'https://www.gutenberg.org/cache/epub/' + str(booknum) + '/pg' + str(booknum) + '.txt'
        filename_temp = 'pg' + str(booknum) + '.txt'
      if verbose:
        print('Trying: ', url)
      try:
        if verbose:
          path_to_file_temp = tf.keras.utils.get_file(filename_temp, url)
        else:
          with contextlib.redirect_stdout(download_log):
            path_to_file_temp = tf.keras.utils.get_file(filename_temp, url)
        temptext = open(path_to_file_temp, 'rb').read().decode(encoding='utf-8')
        tf.io.gfile.remove(path_to_file_temp)
        if (temptext.find('Language: English') >= 0):
          offset = random.randint(-20,20)
          header = 2000
          total_length = 200000
          chopoffend = 10000
          if len(temptext) > (header+total_length+offset+chopoffend):
            foundbook = True
            text_random += temptext[header+offset:header+total_length+offset]
            if verbose:
              print('New size of dataset: ', len(text_random))
          elif len(temptext) > (header+12000):
            foundbook = True
            text_random += temptext[header:-chopoffend]
            if verbose:
              print('New size of dataset: ', len(text_random))
          else:
            if verbose:
              print('Not long enough. Trying again...')
        else:
          if verbose:
            print('Not English. Trying again...')
        del temptext
      except:
        if verbose:
          print('Not valid file. Trying again...')
        foundbook = False
    if verbose:
      print("Found " + str(b+1) + " books so far...")
  del download_log
  return preprocess_text(text_random)

if restart:
  vocab_text = getMyText()


# Make Vocab

vocab_size = 8192
sequence_length = 128

if restart:
  vectorize_layer = TextVectorization(
      standardize='lower',
      split='whitespace',
      max_tokens=8192,
      output_mode='int',
  )

  vectorize_layer.adapt([vocab_text])
  vocabulary = vectorize_layer.get_vocabulary()
  vocab_size = len(vocabulary)

  with open(path + r"\vocabulary.txt", "w", encoding="utf-8") as file:
    for word in vocabulary:
        file.write(word + "\n")

if restart == False:
  with open(path + r"\vocabulary.txt", "r") as file:
      vocabulary = [word.strip() for word in file.readlines()]
      vocabulary = vocabulary

  vectorize_layer = TextVectorization(
      vocabulary=vocabulary,
      standardize='lower',
      split='whitespace',
      max_tokens=vocab_size,
      output_mode='int',
      )
  
print(vocabulary[:20])
print(vocabulary[-20:])



### Turn text into dataset ###

# This function will generate our sequence pairs:
def split_input_target(sequence):
    input_ids = sequence[:-1]
    target_ids = sequence[1:]
    return input_ids, target_ids

# This function will create the dataset
def text_to_dataset(text):
  all_ids = vectorize_layer(text)
  ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
  del all_ids
  sequences = ids_dataset.batch(sequence_length+1, drop_remainder=True)
  del ids_dataset

  # Call the function for every sequence in our list to create a new dataset
  # of input->target pairs
  dataset = sequences.map(split_input_target)
  del sequences
  return dataset


# Test on vocab text

if restart:
  vocab_ds = text_to_dataset(vocab_text)

def text_from_ids(ids):
  text = ''.join([vocabulary[index] for index in ids])
  return postprocess_text(text)

vocabulary_adjusted = vocabulary
vocabulary_adjusted[0] = '[UNK]'
vocabulary_adjusted[1] = ''

words_from_ids = tf.keras.layers.StringLookup(vocabulary=vocabulary_adjusted, invert=True)

if restart:
  for input_example, target_example in vocab_ds.take(1):
    print("Input: ")
    print(input_example)
    print(text_from_ids(input_example))
    print(words_from_ids(input_example))
    print("Target: ")
    print(target_example)
    print(text_from_ids(target_example))

BATCH_SIZE = 64
BUFFER_SIZE = 10000

def setup_dataset(dataset):
  dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))
  return dataset

if restart:
  vocab_ds = setup_dataset(vocab_ds)



### Build Model ###

# Create our custom model. Given a sequence of characters, this
# model's job is to predict what character should come next.
class MarkTwainModel(tf.keras.Model):

  # This is our class constructor method, it will be executed when
  # we first create an instance of the class
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__()

    # Our model will have three layers:

    # 1. An embedding layer that handles the encoding of our vocabulary into
    #    a vector of values suitable for a neural network
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    # 2. A GRU layer that handles the "memory" aspects of our RNN. If you're
    #    wondering why we use GRU instead of LSTM, and whether LSTM is better,
    #    take a look at this article: https://datascience.stackexchange.com/questions/14581/when-to-use-gru-over-lstm
    #    then consider trying out LSTM instead (or in addition to!)
    self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
    self.lstm1 = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True)
    self.lstm2 = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True)
    self.lstm3 = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True)
    self.lstm4 = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True)


    self.hidden1 = tf.keras.layers.Dense(embedding_dim*64, activation='relu')
    self.hidden2 = tf.keras.layers.Dense(embedding_dim*16, activation='relu')
    self.hidden3 = tf.keras.layers.Dense(embedding_dim*4, activation='relu')

    # 3. Our output layer that will give us a set of probabilities for each
    #    character in our vocabulary.
    self.dense = tf.keras.layers.Dense(vocab_size)

  # This function will be executed for each epoch of our training. Here
  # we will manually feed information from one layer of our network to the
  # next.
  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs

    # 1. Feed the inputs into the embedding layer, and tell it if we are
    #    training or predicting
    x = self.embedding(x, training=training)

    # 2. If we don't have any state in memory yet, get the initial random state
    #    from our GRUI layer.
    batch_size = tf.shape(inputs)[0]

    if states is None:
      states1 = [tf.zeros([batch_size, self.lstm1.units]), tf.zeros([batch_size, self.lstm1.units])]
      states2 = [tf.zeros([batch_size, self.lstm2.units]), tf.zeros([batch_size, self.lstm2.units])]
      states3 = [tf.zeros([batch_size, self.lstm3.units]), tf.zeros([batch_size, self.lstm3.units])]
      states4 = [tf.zeros([batch_size, self.lstm4.units]), tf.zeros([batch_size, self.lstm4.units])]
    else:
      states1 = states[0]
      states2 = states[1]
      states3 = states[2]
      states4 = states[3]
    # 3. Now, feed the vectorized input along with the current state of memory
    #    into the gru layer.
    x, state_h_1, state_c_1 = self.lstm1(x, initial_state=states1, training=training)
    states_out_1 = [state_h_1,state_c_1]

    x, state_h_2, state_c_2 = self.lstm2(x, initial_state=states2, training=training)
    states_out_2 = [state_h_2,state_c_2]

    x, state_h_3, state_c_3 = self.lstm3(x, initial_state=states3, training=training)
    states_out_3 = [state_h_3,state_c_3]

    x, state_h_4, state_c_4 = self.lstm4(x, initial_state=states4, training=training)
    states_out_4 = [state_h_4,state_c_4]

    states_out = [states_out_1, states_out_2, states_out_3, states_out_4]
    #states_out = [states_out_1, states_out_2]

    x = self.hidden1(x,training=training)
    x = self.hidden2(x,training=training)
    x = self.hidden3(x,training=training)
    # 4. Finally, pass the results on to the dense layer
    x = self.dense(x, training=training)

    # 5. Return the results
    if return_state:
      return x, states_out
    else:
      return x
    
if restart:
  dataset = vocab_ds
  del vocab_text
  del vocab_ds
else:
  new_text = getRandomText(numbooks = 10)
  dataset = text_to_dataset(new_text)
  del new_text
  dataset = setup_dataset(dataset)

# Create an instance of our model
#vocab_size=len(ids_from_chars.get_vocabulary())
embedding_dim = 128
rnn_units = 512

model = MarkTwainModel(vocab_size, embedding_dim, rnn_units)

# Verify the output of our model is correct by running one sample through
# This will also compile the model for us. This step will take a bit.
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

model.summary()



### Model prep ###

# Here's the code we'll use to sample for us. It has some extra steps to apply
# the temperature to the distribution, and to make sure we don't get empty
# characters in our text. Most importantly, it will keep track of our model
# state for us.

class OneStep(tf.keras.Model):
  def __init__(self, model, vectorize_layer, vocabulary, temperature=1):
    super().__init__()
    self.temperature=temperature
    self.model = model
    self.vectorize_layer = vectorize_layer
    self.vocabulary = vocabulary
    #print("initialized")

    # Create a mask to prevent "" or "[UNK]" from being generated.
    skip_ids = StringLookup(vocabulary=vocabulary, mask_token=None)(['', '[UNK]'])[:, None]
    #print(skip_ids)
    #print("3")
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices = skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(vocabulary)])
    #print("4")
    self.prediction_mask = tf.sparse.to_dense(sparse_mask,validate_indices=False)
    #print("5")

  @tf.function
  def generate_one_step(self, inputs, states=None):
    if states is not None:
        # Enforce consistent shape
        states = [
            [tf.reshape(h, (1, self.model.lstm1.units)), tf.reshape(c, (1, self.model.lstm1.units))]
            for h, c in states
        ]

    input_ids = self.vectorize_layer(inputs)

    # Explicitly set shape to avoid shape mismatches
    input_ids.set_shape([1, None])  # (batch=1, sequence length unknown)

    predicted_logits, states = self.model(inputs=input_ids, states=states, return_state=True)

    predicted_logits = predicted_logits[:, -1, :] / self.temperature
    predicted_logits += self.prediction_mask
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    return words_from_ids(predicted_ids), states
  
def produce_sample(model, vectorize_layer, vocabulary, temp, epoch, prompt):
  # Create an instance of the character generator
  #print("entered")
  one_step_model = OneStep(model, vectorize_layer, vocabulary, temp)
  #print("rand one step")
  # Now, let's generate a 1000 character chapter by giving our model "Chapter 1"
  # as its starting text
  states = None
  next_char = tf.constant([preprocess_text(prompt)])
  result = [tf.constant([prompt])]

  for n in range(1000):
    next_char, states = one_step_model.generate_one_step(next_char, states=states)
    #print(next_char)
    result.append(next_char)
    #print(result)

  result = tf.strings.join(result)
  #print(result)

  # Print the results formatted.
  #print('Temp: ' + str(temp) + '\n')
  print(postprocess_text(result[0].numpy().decode('utf-8')))
  with open(path + r'\tree.txt', 'a', encoding='utf-8') as file:
      print('Epoch: ' + str(epoch) + '\n', file=file)
      print('Temp: ' + str(temp) + '\n', file=file)
      print(postprocess_text(result[0].numpy().decode('utf-8')), file=file)
      print('\n\n', file=file)
  del states
  del next_char
  del result

if restart == False:
  model.load_weights(path + "lstm_gru_SH_modelweights_fall2023-random_urls.h5")

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
opt = tf.keras.optimizers.Adam(learning_rate=0.002, clipnorm=1.0)
model.compile(optimizer=opt, loss=loss)

num_epochs_total = 70
if restart:
  start_epoch = 0
else:
  start_epoch = epoch_to_pickup
for e in range(start_epoch, num_epochs_total):
  success = False
  while not success:
      try:
          print("epoch: ", e)
          new_text = getMyText()
          dataset = text_to_dataset(new_text)
          del new_text
          dataset = setup_dataset(dataset)
          
          model.optimizer.learning_rate.assign(0.002)
          model.fit(dataset, epochs=1, verbose=1)
          print("finished training...")
          del dataset
          success = True
      except Exception as ex:
          print(f"Training error: {ex}")
          gc.collect()
          tf.keras.backend.clear_session()
          try:
              del dataset
          except:
              print("dataset already deleted")
          print("retrying epoch: " , e)

try:
    for temp in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        produce_sample(model, vectorize_layer, vocabulary, temp, e, 'The world seemed like such a peaceful place until the magic tree was discovered in London.')
    print("samples produced...")
except Exception as ex:
    print(f"Sampling error (not retraining): {ex}")

gc.collect()
tf.keras.backend.clear_session()
print("session cleared (to save memory)...")

