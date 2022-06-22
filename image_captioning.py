#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf


import matplotlib.pyplot as plt
import collections
import random
import numpy as np
import os
import time
import json
import pandas as pd



# export PATH="/usr/local/cuda-11.2/bin:$PATH" 
# export LD_LIBRARY_PATH="/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH"



stan_data = pd.read_csv("Data/stanford_df_rectified.csv")


img_path = "/home/ubuntu/Data/stanford_img/content/stanford_images/"
stan_data['Image_name'] = stan_data['Image_name'].apply(lambda x: img_path + str(x) + ".jpg")
stan_data['Paragraph'] = stan_data['Paragraph'].apply(lambda x: "<start> " + str(x) + " <end>")

train_df = stan_data[stan_data['train'] == 1]
val_df = stan_data[stan_data['val'] == 1]
test_df = stan_data[stan_data['test'] == 1] 

train_img_name_vector = list(train_df['Image_name'].values)
val_img_name_vector = list(val_df['Image_name'].values)
test_img_name_vector = list(test_df['Image_name'].values)

train_captions = list(train_df['Paragraph'].values)
val_captions = list(val_df['Paragraph'].values)
test_captions = list(test_df['Paragraph'].values)

print(train_captions[0])

#Image.open(train_img_name_vector[0])


### Preprocess the images using InceptionV3
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


### Initialize InceptionV3 and load the pretrained Imagenet weights
image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


### Caching the features extracted from InceptionV3
from tqdm import tqdm

#Get unique images
encode_train = sorted(set(train_img_name_vector))

image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(
  load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(16)

for img, path in tqdm(image_dataset):
  batch_features = image_features_extract_model(img)
  batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1, batch_features.shape[3]))

  for bf, p in zip(batch_features, path):
    path_of_feature = p.numpy().decode("utf-8")
    np.save(path_of_feature, bf.numpy())


# ## Preprocess and tokenize the captions
# * Tokenize all captions by mapping each word to it's index in the vocabulary. All output sequences will be padded to length 50.
# * Create word-to-index and index-to-word mappings to display results.

caption_dataset = tf.data.Dataset.from_tensor_slices(train_captions)

# We will override the default standardization of TextVectorization to preserve
# "<>" characters, so we preserve the tokens for the <start> and <end>.
def standardize(inputs):
  inputs = tf.strings.lower(inputs)
  return tf.strings.regex_replace(inputs,
                                  r"!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~", "")

# Max word count for a caption.
max_length = 50
# Use the top 5000 words for a vocabulary.
vocabulary_size = 5000
tokenizer = tf.keras.layers.TextVectorization(
    max_tokens=vocabulary_size,
    standardize=standardize,
    output_sequence_length=max_length)
# Learn the vocabulary from the caption data.
tokenizer.adapt(caption_dataset)


# Create the tokenized vectors
cap_vector = caption_dataset.map(lambda x: tokenizer(x))


# Create mappings for words to indices and indicies to words.
word_to_index = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary())
index_to_word = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary(),
    invert=True)


# ## Split the data into training and testing

train_img_to_cap_vector = collections.defaultdict(list)
cap_train = []
cap_val = []
cap_test = []
vcaption_dataset = tf.data.Dataset.from_tensor_slices(val_captions)
vcap_vector = vcaption_dataset.map(lambda x: tokenizer(x))

tcaption_dataset = tf.data.Dataset.from_tensor_slices(test_captions)
tcap_vector = tcaption_dataset.map(lambda x: tokenizer(x))

for img, cap in zip(train_img_name_vector, cap_vector):
  cap_train.append(cap)

for img, cap in zip(val_img_name_vector, vcap_vector):
  cap_val.append(cap)

for img, cap in zip(test_img_name_vector, tcap_vector):
  cap_test.append(cap)



img_name_train = train_img_name_vector
# cap_train = img_to_cap_vector.values()

img_name_val = val_img_name_vector
# cap_val = val_captions

img_name_test = test_img_name_vector
# cap_test = test_captions




len(img_name_train), len(cap_train)



BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
num_steps = len(img_name_train) // BATCH_SIZE
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64



# Load the numpy files
def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8')+'.npy')
    return img_tensor, cap


dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int64]),
          num_parallel_calls=tf.data.AUTOTUNE)

# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder = True)
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)




class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # attention_hidden_layer shape == (batch_size, 64, units)
    attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                         self.W2(hidden_with_time_axis)))

    # score shape == (batch_size, 64, 1)
    score = self.V(attention_hidden_layer)

    # attention_weights shape == (batch_size, 64, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        #print(len(x[0]))
        x = tf.nn.relu(x)
        return x



class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    # defining attention as a separate model
    context_vector, attention_weights = self.attention(features, hidden)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))



encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, tokenizer.vocabulary_size())



optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


# ## Checkpoint

# In[36]:


# checkpoint_path = "./checkpoints/train"
# ckpt = tf.train.Checkpoint(encoder=encoder,
#                            decoder=decoder,
#                            optimizer=optimizer)
# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)


# In[37]:


start_epoch = 0
# if ckpt_manager.latest_checkpoint:
#   start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
#   # restoring the latest checkpoint in checkpoint_path
#   ckpt.restore(ckpt_manager.latest_checkpoint)


loss_plot = []


@tf.function
def train_step(img_tensor, target):
  loss = 0
  
  # initializing the hidden state for each batch
  hidden = decoder.reset_state(batch_size=target.shape[0])

  dec_input = tf.expand_dims([word_to_index(tf.convert_to_tensor('<start>'))] * target.shape[0], 1)

  with tf.GradientTape() as tape:
      features = encoder(img_tensor)
      
      for i in range(1, target.shape[1]):
          # passing the features through the decoder
          predictions, hidden, _ = decoder(dec_input, features, hidden)

          loss += loss_function(target[:, i], predictions)

         
          dec_input = tf.expand_dims(target[:, i], 1)

  total_loss = (loss / int(target.shape[1]))

  trainable_variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, trainable_variables)

  optimizer.apply_gradients(zip(gradients, trainable_variables))

  return loss, total_loss



EPOCHS = 100
for epoch in range(0, EPOCHS):
    start = time.time()
    total_loss = 0
    
    for (batch, (img_tensor, target)) in enumerate(dataset):
        
        
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss

        if batch % 10 == 0:
            average_batch_loss = batch_loss.numpy()/int(target.shape[1])
            print(f'Epoch {epoch+1} Batch {batch} Loss {average_batch_loss:.4f}')
    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / num_steps)

#     if epoch % 5 == 0:
#       ckpt_manager.save()

    print(f'Epoch {epoch+1} Loss {total_loss/num_steps:.6f}')
    print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')



plt.plot(loss_plot)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.show()


def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                 -1,
                                                 img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([word_to_index(tf.convert_to_tensor('<start>'))], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input,
                                                         features,
                                                         hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        predicted_word = tf.compat.as_text(index_to_word(tf.convert_to_tensor(predicted_id)).numpy())
        result.append(predicted_word)

        if predicted_word == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot



def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(100, 100))

    len_result = len(result)
    for i in range(len_result):
        temp_att = np.resize(attention_plot[i], (8, 8))
        grid_size = max(int(np.ceil(len_result/2)), 2)
        ax = fig.add_subplot(grid_size, grid_size, i+1)
        ax.set_title(result[i])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())
    
    fig.canvas.draw()
    plt.tight_layout()
    plt.show()






rid = np.random.randint(0, len(img_name_val))
image = img_name_val[rid]

real_caption = ' '.join([tf.compat.as_text(index_to_word(i).numpy())
                         for i in cap_val[rid] if i not in [0]])
result, attention_plot = evaluate(image)

print('Real Caption:', real_caption)
print('Prediction Caption:', ' '.join(result))



import nltk



img_ids = []
#real_caps = []
for i in range(len(img_name_test)):
    rid = i 
    if i%100 == 0:
        print(i)
    image = img_name_test[rid]

    #real_caption = ' '.join([tf.compat.as_text(index_to_word(k).numpy())for k in cap_test[rid] if k not in [0]])
   
    
    result, attention_plot = evaluate(image)
    pred_caption = ' '.join(result)
    
    img_ids.append(image)
    #real_caps.append(real_caption)
    data.append(pred_caption)
    #bscore = nltk.translate.bleu_score.sentence_bleu([real_caption.split()], pred_caption.split())
    #print("BLEU Score:",bscore )

d = {'id': img_ids, 'pred_caption': data}
test_res_df = pd.DataFrame(d)



test_res_df.to_csv("test_captions_nosent.csv")





import nltk
nltk.download('vader_lexicon')
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# function to print sentiments of the sentence.
def sentiment_scores(sentence):

    sid_obj = SentimentIntensityAnalyzer()

    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(sentence)

    if sentiment_dict['compound'] >= 0.05 :
        print("Positive")
        return 1

    elif sentiment_dict['compound'] <= - 0.05 :
        print("Negative")
        return -1

    else :
        print("Neutral")
        return 0
 


    
test_res_df['sentiment'] = test_res_df['pred_caption'].apply(lambda x: sentiment_scores(x))
    


# from translate import Translator
# # translator= Translator(from_lang="english",to_lang="french")
# # translation = translator.translate("How are you doing?")
# # print(translation)

# test_res_df['Spanish'] = test_res_df['pred_caption'].apply(lambda x : Translator(from_lang="english",to_lang="spanish").translate(x))
# test_res_df['Italian'] = test_res_df['pred_caption'].apply(lambda x : Translator(from_lang="english",to_lang="italian").translate(x))
# test_res_df['French'] = test_res_df['pred_caption'].apply(lambda x : str(Translator(from_lang="english",to_lang="french").translate(x)))
test_res_df.to_csv("test_captions.csv")



    


