import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import pickle
import string
import re
from log import logger

# Load contraction expansion data
with open("Data/Input/contraction_expansion.txt", 'rb') as fp:
    contractions = pickle.load(fp)

def english_preprocessing(data, col):
    """
    * method: english_preprocessing
    * description: Preprocesses English text in a DataFrame column by converting to lowercase, removing non-alphabetic characters, and adjusting spaces.
    * return: DataFrame with the specified column preprocessed
    *
    * who             when           version  change (include bug# if apply)
    * ----------      -----------    -------  ------------------------------
    * Shubham M      05-SEPT-2024    1.0      initial creation
    *
    * Parameters
    *   data (pd.DataFrame): DataFrame containing the text data.
    *   col (str): Column name containing the text data to preprocess.
    """
    data[col] = data[col].astype(str)
    data[col] = data[col].apply(lambda x: x.lower())  # Convert to lowercase
    data[col] = data[col].apply(lambda x: re.sub("[^A-Za-z\s]", "", x))  # Remove non-alphabetic characters
    data[col] = data[col].apply(lambda x: x.replace("\s+", " "))  # Adjust multiple spaces
    data[col] = data[col].apply(lambda x: " ".join([word for word in x.split()]))  # Remove extra spaces
    return data

def marathi_preprocessing(data, col):
    """
    * method: marathi_preprocessing
    * description: Preprocesses Marathi text in a DataFrame column by converting to lowercase, removing digits, punctuation, and English characters.
    * return: DataFrame with the specified column preprocessed
    *
    * who             when           version  change (include bug# if apply)
    * ----------      -----------    -------  ------------------------------
    * Shubham M      05-SEPT-2024    1.0      initial creation
    *
    * Parameters
    *   data (pd.DataFrame): DataFrame containing the text data.
    *   col (str): Column name containing the text data to preprocess.
    """
    data[col] = data[col].apply(lambda x: x.lower())  # Convert to lowercase
    data[col] = data[col].apply(lambda x: re.sub(r'\d', '', x))  # Remove digits
    data[col] = data[col].apply(lambda x: re.sub(r'\s+', ' ', x))  # Adjust multiple spaces
    data[col] = data[col].apply(lambda x: re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,।]", "", x))  # Remove punctuation and special characters
    data[col] = data[col].apply(lambda x: x.strip())  # Remove leading/trailing spaces
    data[col] = data[col].apply(lambda x: re.sub('[a-zA-Z]', '', x))  # Remove English alphabetic characters
    return data

def expand_contras(text):
    """
    * method: expand_contras
    * description: Expands contractions in English text based on a predefined dictionary.
    * return: Text with contractions expanded
    *
    * who             when           version  change (include bug# if apply)
    * ----------      -----------    -------  ------------------------------
    * Shubham M      05-SEPT-2024    1.0      initial creation
    *
    * Parameters
    *   text (str or list): Input text or list of texts where contractions will be expanded.
    """
    if isinstance(text, str):
        for key in contractions:
            value = contractions[key]
            text = text.replace(key, value)
        return text
    else:
        return text

def preprocessing(input_data_path):
    """
    * method: preprocessing
    * description: Loads data, preprocesses English and Marathi columns, expands contractions, and saves the cleaned data.
    * return: None
    *
    * who             when           version  change (include bug# if apply)
    * ----------      -----------    -------  ------------------------------
    * Shubham M      05-SEPT-2024    1.0      initial creation
    *
    * Parameters
    *   input_data_path (str): Path to the input data file.
    """
    # Load data
    df = pd.read_table(input_data_path, encoding='utf-8', names=['English', 'Marathi', 'Attribution'])
    logger.info(f"Input data loaded: {df.shape}")
    
    # Drop unused column
    df.drop(['Attribution'], axis=1, inplace=True)
    
    # Preprocess data
    df = english_preprocessing(df, 'English')
    df = marathi_preprocessing(df, 'Marathi')
    
    # Expand contractions in English
    df['English'] = df['English'].apply(lambda x: expand_contras(x))

    # Save cleaned data
    df.to_csv("Data/Output/english_marathi_data_clean.csv", index=None)
    logger.info("Output file saved at: Data/Output/english_marathi_data_clean.csv")

if __name__ == '__main__':
    # Entry point for script execution
    input_data_path = 'Data/Input/mar.txt'
    preprocessing(input_data_path)
