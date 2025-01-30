import pandas as pd
import helper_functions as hf

# Putting training data in df_train
df_train = pd.read_csv(r'C:\Users\User\Desktop\Python\CNN_DailyMail\Data\train.csv')
#print(df_train.head()) # First 5 rows
#print(df_train.isnull().sum()) # No null rows

# FIRST WE WILL DO THE SENTENCE TOKENIZATION AND SUMMARIZATION FROM THAT
articles = df_train['article'].head().tolist() # Converting column to list so we can preprocess
#print(articles)
articles = hf.remove_whitespaces(articles) # Removing whitespace

articles = hf.remove_stopwords(articles) # Removing stopwords
print(articles[0])
articles = hf.lemmatize_text(articles) # Lemmatize text
print(articles[0])
articles = hf.nltk_sentence_tokenizer(articles) # Tokenizing sentences


for i in range(len(articles)):
    articles[i] = hf.remove_non_word_characters(articles[i]) # Removing non word chars from every sentence

#print(articles)