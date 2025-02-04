import re
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Removing non-word characters - RETURNS DATAFRAME OR LIST
def remove_non_word_characters(data):
    if isinstance(data, pd.DataFrame):
        return data.str.replace(r"[^\w\s$%]", "", regex=True)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = re.sub(r"[^\w\s$%]", "", data[i])
            print(data[i])
        return data


# Removing whitespaces - RETURNS DATAFRAME OR LIST
def remove_whitespaces(data):
    if isinstance(data, pd.DataFrame):
        return data.str.replace(r"[^\s+]", " ", regex=True)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = re.sub(r"\s+", " ", data[i])
            #print(data[i])
        return data
    
# NLTK sentence tokenizer - RETURNS ONLY A LIST
def nltk_sentence_tokenizer(data):
    if isinstance(data, pd.DataFrame):
        print(data)
        return data.apply(sent_tokenize)
    elif isinstance(data, list):
        return [sent_tokenize(d) for d in data]
    
# Removing stopwords - RETURNS A LIST
def remove_stopwords(data):
    if isinstance(data, pd.DataFrame):
        return " ".join(word for word in data.split() if word.lower() not in stop_words)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = " ".join(word for word in data[i].split() if word.lower() not in stop_words)
        return data
        
    
# Text lemmatization - RETURNS A LIST
def lemmatize_text(data):
    if isinstance(data, pd.DataFrame):
        return " ".join(word for word in data.split() if word.lower() not in stop_words)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = " ".join(lemmatizer.lemmatize(word) for word in data[i].split())
        return data