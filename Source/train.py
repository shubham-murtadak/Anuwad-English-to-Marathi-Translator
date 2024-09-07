import os
# Suppress INFO and WARNING logs from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import re
import time
import string
import pickle
import warnings
import pandas as pd
import numpy as np
import googletrans
import tensorflow as tf


from flask import Flask, render_template, request, make_response
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding, Concatenate, Dropout
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from googletrans import Translator
from deep_translator import GoogleTranslator
from gtts import gTTS
from IPython.display import Audio
from Source.log import logger
# import tensorflow as tf
from tensorflow.python.keras import backend as K
from Source.preprocess import english_preprocessing, expand_contras


# Ignore warnings
warnings.filterwarnings("ignore")


# Define encoder and decoder models as global variables
encoder_model = None
decoder_model = None

# # Create and fit tokenizer
tokenizer = Tokenizer()
df = pd.read_csv('Data\Output\english_marathi_data_clean.csv')
tokenizer.fit_on_texts(df['English'])



hparams = {
    "batch_size": 128,
    "cnn_filter_sizes": [128, 128, 128],
    "cnn_kernel_sizes": [5, 5, 5],
    "cnn_pooling_sizes": [5, 5, 40],
    "constraint_learning_rate": 0.01,
    "embedding_dim": 100,
    "embedding_trainable": False,
    "learning_rate": 0.005,
    "max_num_words": 10000,
    "max_sequence_length": 250
}


# # marathi words index mapping
mar_index_word = dict()
with open('Data/Other_data/mar_index_word.pkl', "rb") as f:
    mar_index_word = pickle.load(f)

mar_word_index = dict()
with open("Data/Other_data/mar_word_index.pkl", "rb") as f:
    mar_word_index = pickle.load(f)

# # english words index mapping
eng_index_word = dict()
with open('Data/Other_data/eng_index_word.pkl', 'rb') as f:
    eng_index_word = pickle.load(f)

eng_word_indec = dict()
with open("Data/Other_data/eng_word_indec.pkl", "rb") as f:
    eng_word_indec = pickle.load(f)


# # english contraction expansion
with open("Data\Input\contraction_expansion.txt", 'rb') as fp:
    contractions = pickle.load(fp)





def tokenize_sent(text):
  '''
  Take list on texts as input and
  returns its tokenizer and enocded text
  '''
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(text)

  return tokenizer, tokenizer.texts_to_sequences(text)


class AttentionLayer(tf.keras.layers.Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs

        logger.debug(f"encoder_out_seq.shape = {encoder_out_seq.shape}")
        logger.debug(f"decoder_out_seq.shape = {decoder_out_seq.shape}")

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state
            inputs: (batchsize * 1 * de_in_dim)
            states: (batchsize * 1 * de_latent_dim)
            """

            logger.debug("Running energy computation step")

            if not isinstance(states, (list, tuple)):
                raise TypeError(f"States must be an iterable. Got {states} of type {type(states)}")

            encoder_full_seq = states[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch size * en_seq_len * latent_dim
            W_a_dot_s = K.dot(encoder_full_seq, self.W_a)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim

            logger.debug(f"U_a_dot_h.shape = {U_a_dot_h.shape}")

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            Ws_plus_Uh = K.tanh(W_a_dot_s + U_a_dot_h)

            logger.debug(f"Ws_plus_Uh.shape = {Ws_plus_Uh.shape}")

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.squeeze(K.dot(Ws_plus_Uh, self.V_a), axis=-1)
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            logger.debug(f"ei.shape = {e_i.shape}")

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """

            logger.debug("Running attention vector computation step")

            if not isinstance(states, (list, tuple)):
                raise TypeError(f"States must be an iterable. Got {states} of type {type(states)}")

            encoder_full_seq = states[-1]

            # <= batch_size, hidden_size
            c_i = K.sum(encoder_full_seq * K.expand_dims(inputs, -1), axis=1)

            logger.debug(f"ci.shape = {c_i.shape}")

            return c_i, [c_i]

        # we don't maintain states between steps when computing attention
        # attention is stateless, so we're passing a fake state for RNN step function
        fake_state_c = K.sum(encoder_out_seq, axis=1)
        fake_state_e = K.sum(encoder_out_seq, axis=2)  # <= (batch_size, enc_seq_len, latent_dim

        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e], constants=[encoder_out_seq]
        )

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c], constants=[encoder_out_seq]
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]




# def expand_contras(text):
#     '''
#     takes input as word or list of words
#     if it is string and contracted it will expand it
#     example:
#     it's --> it is
#     won't --> would not
#     '''
#     if type(text) is str:
#         for key in contractions:
#             value = contractions[key]
#             text = text.replace(key, value)
#         return text
#     else:
#         return text
    
    

# # sentence pre-processing before tokenization
# def english_preprocessing(data, col):
#     data[col] = data[col].astype(str)
#     data[col] = data[col].apply(lambda x: x.lower())
#     data[col] = data[col].apply(lambda x: re.sub("[^A-Za-z\s]", "", x))  # all non-alphabetic characters

#     data[col] = data[col].apply(lambda x: x.replace("\s+", " "))  # adjust multiple spaces
#     data[col] = data[col].apply(lambda x: " ".join([word for word in x.split()]))

#     return data




# def domain():

df_path='Data\Output\english_marathi_data_clean.csv'
df = pd.read_csv(df_path)
logger.info("Data file loaded !")
print("shape :",df.shape)
logger.info(f"shape of data :{df.shape}")


#Adding start and end tokens to target sentecnes
df['Marathi'] =df.Marathi.apply(lambda x: 'sos '+ x + ' eos')

#convet english and marathi columns to list
eng_texts = df.English.to_list()
mar_texts = df.Marathi.to_list()


# Tokenize english and marathi sentences
eng_tokenizer, eng_encoded= tokenize_sent(text= eng_texts)
mar_tokenizer, mar_encoded= tokenize_sent(text= mar_texts)

logger.info(f"TOkenization done : {eng_encoded[100], mar_encoded[100]}")

eng_index_word = eng_tokenizer.index_word
eng_word_indec= eng_tokenizer.word_index

# logger.info(f"{print(list(eng_index_word.items())[:5]}")
ENG_VOCAB_SIZE = len(eng_tokenizer.word_counts)+1
logger.info(f"Enlgish Vocab size calculated :{ENG_VOCAB_SIZE}")

mar_index_word = mar_tokenizer.index_word
mar_word_index= mar_tokenizer.word_index

MAR_VOCAB_SIZE=len(mar_tokenizer.word_counts)+1
logger.info(f"Marathi Vocab size calculated :{MAR_VOCAB_SIZE}")

# Padding Making input sentences as max length of input sentence with padding zero
max_eng_len = max(len(seq) for seq in eng_encoded)
logger.info(f"Max English length :{max_eng_len}")

max_mar_len = max(len(seq) for seq in mar_encoded)
logger.info(f"Max Marathi length :{max_mar_len}")

eng_padded = pad_sequences(eng_encoded, maxlen=max_eng_len, padding='post')
mar_padded = pad_sequences(mar_encoded, maxlen=max_mar_len, padding='post')

logger.info("English padded shape: %s", eng_padded.shape)
logger.info("Marathi padded shape: %s", mar_padded.shape)

eng_padded= np.array(eng_padded)
mar_padded= np.array(mar_padded)

#Data Splitting
X_train, X_test, y_train, y_test = train_test_split(eng_padded, mar_padded, test_size=0.1, random_state=0)
logger.info("Data splitting done !shapes - X_train: %s, X_test: %s, y_train: %s, y_test: %s", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Encoder
encoder_inputs = Input(shape=(max_eng_len,))
enc_emb = Embedding(ENG_VOCAB_SIZE, 1024)(encoder_inputs)

# Bidirectional lstm layer
enc_lstm1 = Bidirectional(LSTM(256,return_sequences=True,return_state=True))
encoder_outputs1, forw_state_h, forw_state_c, back_state_h, back_state_c = enc_lstm1(enc_emb)

final_enc_h = Concatenate()([forw_state_h,back_state_h])
final_enc_c = Concatenate()([forw_state_c,back_state_c])

encoder_states =[final_enc_h, final_enc_c]

# Set up the decoder.
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(MAR_VOCAB_SIZE, 1024)
dec_emb = dec_emb_layer(decoder_inputs)
#LSTM using encoder_states as initial state
decoder_lstm = LSTM(512, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

#Attention Layer
attention_layer = AttentionLayer(name="attension_layer")
attention_result, attention_weights = attention_layer([encoder_outputs1, decoder_outputs])

# Concat attention output and decoder LSTM output
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attention_result])

#Dense layer
decoder_dense = Dense(MAR_VOCAB_SIZE, activation='softmax')
decoder_outputs = decoder_dense(decoder_concat_input)


# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

logger.info("Model created successfully !")
#   print(model.summary())

#   # Compile model
#   model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#   # Define the checkpoint callback with a complete file path
#   checkpoint = ModelCheckpoint(
#       filepath='/kaggle/working/model_checkpoint.keras',  # Simplified path for testing
#       monitor='val_accuracy',
#       save_best_only=True,
#       mode='max',
#       verbose=1
#   )

#   # Define early stopping
#   early_stopping = EarlyStopping(
#       monitor='val_accuracy',
#       patience=5,
#       mode='max',
#       verbose=1
#   )

#   # List of callbacks
#   callbacks_list = [checkpoint, early_stopping]

#   # Data preparation
#   encoder_input_data = X_train
#   decoder_input_data = y_train[:, :-1]
#   decoder_target_data = y_train[:, 1:]

#   encoder_input_test = X_test
#   decoder_input_test = y_test[:, :-1]
#   decoder_target_test = y_test[:, 1:]

#   # Training
#   EPOCHS = 50

#   history = model.fit(
#       [encoder_input_data, decoder_input_data],
#       decoder_target_data,
#       epochs=EPOCHS,
#       batch_size=128,
#       validation_data=([encoder_input_test, decoder_input_test], decoder_target_test),
#       callbacks=callbacks_list
#   )

model.load_weights("Data\weights\model.weights.h5")
logger.info("Model loaded successfully !")

#interference model
encoder_model = Model(encoder_inputs, outputs = [encoder_outputs1, final_enc_h, final_enc_c])

decoder_state_h = Input(shape=(512,))
decoder_state_c = Input(shape=(512,))
decoder_hidden_state_input = Input(shape=(34,512))

dec_states = [decoder_state_h, decoder_state_c]

dec_emb2 = dec_emb_layer(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=dec_states)

# Attention inference
attention_result_inf, attention_weights_inf = attention_layer([decoder_hidden_state_input, decoder_outputs2])

decoder_concat_input_inf = Concatenate(axis=-1, name='concat_layer')([decoder_outputs2, attention_result_inf])

dec_states2= [state_h2, state_c2]

decoder_outputs2 = decoder_dense(decoder_concat_input_inf)

decoder_model= Model([decoder_inputs] + [decoder_hidden_state_input, decoder_state_h, decoder_state_c],[decoder_outputs2]+ dec_states2)

logger.info("Encoder decoer model is redy !")




def get_predicted_sentence(input_seq):
    # Encode the input as state vectors.
    enc_output, enc_h, enc_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))

    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = mar_word_index['sos']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [enc_output, enc_h, enc_c ])
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index == 0:
          break
        else:
            # convert max index number to marathi word
            sampled_char = mar_index_word[sampled_token_index]

        if (sampled_char!='end'):
            # aapend it ti decoded sent
            decoded_sentence += ' '+sampled_char

        # Exit condition: either hit max length or find stop token.
        if (sampled_char == 'eos' or len(decoded_sentence.split()) >= 36):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index
        # Update states
        enc_h, enc_c = h, c

    print(decoded_sentence,len(decoded_sentence))
    return decoded_sentence[:-3]


# def get_marathi_sentence(input_sequence):
#     sentence =''
#     for i in input_sequence:
#       if i!=0 :
#         sentence =sentence +mar_index_word[i]+' '
#     return sentence

# def get_english_sentence(input_sequence):
#     sentence =''
#     for i in input_sequence:
#       if i!=0:
#         sentence =sentence +eng_index_word[i]+' '
#     return sentence



if __name__=='__main__':
    # domain()
#   print("jay hanuman dada")
#   df_path='Data\Output\english_marathi_data_clean.csv'
  
#   app.run(debug=True, threaded=True)
  
#   df=pd.read_csv(df_path)
#   print(df.columns)
  # tokenizer = text.Tokenizer(num_words=hparams["max_num_words"])
  # tokenizer.fit_on_texts(df['English'])

  app.run(debug=True, port=5001)
#   # Assuming you have a fitted tokenizer
#   input_text = "i wanted to see you too "
#   input_text = str(input_text)

#   # Convert to DataFrame and preprocess
#   df1 = pd.DataFrame()
#   df1["English"] = [input_text]
#   df1 = english_preprocessing(df1, "English")
#   df1.English = df1.English.apply(lambda x: expand_contras(x))
#   input_text = df1.English.to_list()
#   # input_text = input_text[0]

#   # Tokenize and pad sequences
#   eng_encoded = tokenizer.texts_to_sequences(input_text)  # Note: Pass a list of texts
#   eng_padded = pad_sequences(eng_encoded, maxlen=34, padding='post')

#   print("Encoded:", eng_encoded)
#   print("Padded:", eng_padded)

#   translated = get_predicted_sentence(eng_padded)
#   logger.info("Translation Done !")

#   print(translated)
