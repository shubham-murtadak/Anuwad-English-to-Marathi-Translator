import re
import pickle


# english contraction expansion
with open("data/contraction_expansion.txt", 'rb') as fp:
    contractions= pickle.load(fp)

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

# sentence pre-processing before tokenization
def english_preprocessing(data, col):
    data[col] = data[col].astype(str)
    data[col] = data[col].apply(lambda x: x.lower())
    data[col] = data[col].apply(lambda x: re.sub("[^A-Za-z\s]", "", x))  # all non-alphabetic characters

    data[col] = data[col].apply(lambda x: x.replace("\s+", " "))  # adjust multiple spaces
    data[col] = data[col].apply(lambda x: " ".join([word for word in x.split()]))

    return data