import pandas as pd
import helper_functions as hf
import text_vectorization as tv
import summary_generation as sg


# Putting training data in df_train
df_train = pd.read_csv(r'C:\Users\User\Desktop\Python\CNN_DailyMail\Data\train.csv')
#print(df_train.head()) # First 5 rows
#print(df_train.isnull().sum()) # No null rows

# FIRST WE WILL DO THE SENTENCE TOKENIZATION AND SUMMARIZATION FROM THAT
articles_train = df_train['article'].head().tolist() # Converting column to list so we can preprocess
highlights_train = df_train['highlights'].head().tolist()
#print(articles)
articles_train = hf.remove_whitespaces(articles_train) # Removing whitespace
articles_train = hf.remove_stopwords(articles_train) # Removing stopwords
articles_train = hf.lemmatize_text(articles_train) # Lemmatize text
articles_train = hf.nltk_sentence_tokenizer(articles_train) # Tokenizing sentences
for i in range(len(articles_train)):
    articles_train[i] = hf.remove_non_word_characters(articles_train[i]) # Removing non word chars from every sentence

highlights_train = hf.remove_whitespaces(highlights_train) # Removing whitespace
highlights_train = hf.remove_stopwords(highlights_train) # Removing stopwords
highlights_train = hf.lemmatize_text(highlights_train) # Lemmatize text
highlights_train = hf.nltk_sentence_tokenizer(highlights_train) # Tokenizing sentences
for i in range(len(highlights_train)):
    highlights_train[i] = hf.remove_non_word_characters(highlights_train[i]) # Removing non word chars from every sentence
#print(articles)

# FEATURE EXTRACTION - TEXT VECTORIZATION

# TF-IDF FIRST
tfidf_vectors_articles_train = [] # LIST FOR VECTORS FOR TFIDF METHOD FOR TRAIN ARTICLES
tfidf_vectors_highlights_train = [] # LIST FOR VECTORS FOR TFIDF METHOD FOR HIGHLIGHTS ARTICLES
for i in range(len(articles_train)):
    tfidf_vectors_articles_train.append(tv.tfidf(articles_train[i]))
    tfidf_vectors_highlights_train.append(tv.tfidf(highlights_train[i]))

# WORD EMBEDDINGS



# SIMILARITY SCORING



# MODEL MAKING
