import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding,LSTM, Bidirectional
from tensorflow.keras.models import Sequential

url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
urllib.request.urlretrieve(url,'sarcasm.json')

tf.Configproto()
_truncating = 'post'
_padding = 'post'
_maxlen = 120
embedding_dim = 16
with open('sarcasm.json','r') as f:
    data = json.load(f)

sentences = []
labels = []

for d in data:
    sentences.append(d['headline'])
    labels.append(d['is_sarcastic'])

train_ratio = 0.8
train_size = int(len(data) * train_ratio)

train_sentences = sentences[:train_size]
valid_sentences = sentences[train_size:]

train_labels = labels[:train_size]
valid_labels = labels[train_size:]
vocab_size = 5000
token = Tokenizer(num_words = vocab_size, oov_token='<OOV>')

token.fit_on_texts(sentences)
word_index = token.word_index
train_sequences = token.texts_to_sequences(train_sentences)
valid_sequences = token.texts_to_sequences(valid_sentences)
train_padded = pad_sequences(train_sequences, truncating=_truncating, padding=_padding, maxlen=_maxlen)
valid_padded = pad_sequences(valid_sequences, truncating=_truncating, padding=_padding, maxlen=_maxlen)
train_labels = np.asarray(train_labels)
valid_labels = np.asarray(valid_labels)

model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=_maxlen),
        Bidirectional(LSTM(64)),
        Dense(24, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

checkpoint_path = 'best_performed_model.ckpt'
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                save_weights_only=True,
                                                save_best_only=True,
                                                monitor='val_loss',
                                                verbose=1)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_padded, train_labels,
                    validation_data=(valid_padded, valid_labels),
                    callbacks=[checkpoint],
                    epochs=10000,
                    verbose=2)

model.load_weights(checkpoint_path)