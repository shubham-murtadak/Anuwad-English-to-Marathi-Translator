import os
# Suppress INFO and WARNING logs from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pickle
import time
import pandas as pd
import re
import googletrans
from googletrans import Translator
from deep_translator import GoogleTranslator
from gtts import gTTS
from IPython.display import Audio
from flask import Flask, render_template, request, make_response
from tensorflow.keras.preprocessing.sequence import pad_sequences

from Source.train import get_predicted_sentence
from tensorflow.keras.preprocessing.text import Tokenizer


# # Create and fit tokenizer
tokenizer = Tokenizer()
df = pd.read_csv('Data\Output\english_marathi_data_clean.csv')
tokenizer.fit_on_texts(df['English'])



# # english contraction expansion
with open("Data\Input\contraction_expansion.txt", 'rb') as fp:
    contractions = pickle.load(fp)

# sentence pre-processing before tokenization
def english_preprocessing(data, col):
    data[col] = data[col].astype(str)
    data[col] = data[col].apply(lambda x: x.lower())
    data[col] = data[col].apply(lambda x: re.sub("[^A-Za-z\s]", "", x))  # all non-alphabetic characters

    data[col] = data[col].apply(lambda x: x.replace("\s+", " "))  # adjust multiple spaces
    data[col] = data[col].apply(lambda x: " ".join([word for word in x.split()]))

    return data
  

def expand_contras(text):
    '''
    takes input as word or list of words
    if it is string and contracted it will expand it
    example:
    it's --> it is
    won't --> would not
    '''
    if type(text) is str:
        for key in contractions:
            value = contractions[key]
            text = text.replace(key, value)
        return text
    else:
        return text


app = Flask(__name__)

language_codes = googletrans.LANGUAGES
# Define Marathi as the default language
default_language = 'mr'

# Optionally, create a list of languages including only Marathi
languages = [{"code": default_language, "name": language_codes[default_language]}]


@app.route("/", methods=["GET", "POST"])
def translate():
    if request.method == "POST":
        input_text = request.form.get("input_text")
        input_text1=input_text
        target_language = request.form.get("target_language")
        print("input_text:",input_text)
        # Convert to DataFrame and preprocess
        df1 = pd.DataFrame()
        df1["English"] = [input_text]
        df1 = english_preprocessing(df1, "English")
        df1.English = df1.English.apply(lambda x: expand_contras(x))
        input_text = df1.English.to_list()
  
        # Tokenize and pad sequences
        eng_encoded = tokenizer.texts_to_sequences(input_text)  # Note: Pass a list of texts
        eng_padded = pad_sequences(eng_encoded, maxlen=34, padding='post')
        translated_text= get_predicted_sentence(eng_padded)
        
        # translated_text =GoogleTranslator(source='auto', target=target_language).translate(input_text)
        # translated_text = translate_text(input_text, dest=target_language)
        
        timestamp = int(time.time())
        filename = f"static/op_{timestamp}.mp3"  
        tts = gTTS(translated_text, lang=target_language)
        tts.save(filename)  
        return render_template("index.html", languages=languages, input_text=input_text1, translated_text=translated_text, audio_filename=filename)
      
      
    return render_template("index.html", languages=languages)



if __name__=='__main__':
  app.run(debug=True, port=5000)
#   paste this :http://127.0.0.1:5000/