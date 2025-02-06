
import time
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import normalize
from torch import cosine_similarity
import torch
import helper_functions as hf
import text_vectorization as tv
import summary_generation as sg
from numpy.linalg import norm

np.set_printoptions(threshold=np.inf)  # Print entire array without truncation
pd.set_option("display.width", 300)



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

# TF-IDF

documents_articles = [doc[i] for doc in articles_train for i in range(len(doc))] # All sentences from every article together
documents_highlights = [doc[i] for doc in highlights_train for i in range(len(doc))] # All sentences from every highlights together
#articles_train_tfidf = tv.tfidf(articles_train, documents_articles)
#highlights_train_tfidf = tv.tfidf(highlights_train, documents_highlights)
articles_train_tfidf, highlights_train_tfidf = tv.tfidf(articles_train, documents_articles, highlights_train, documents_highlights)

cosine = []
for i in range(len(articles_train_tfidf)):
    article_matrix = articles_train_tfidf[i]  # Shape: (num_article_sentences, num_features)
    summary_matrix = highlights_train_tfidf[i]  # Shape: (num_summary_sentences, num_features)
    #print(article_matrix)
    #print(summary_matrix)
    article_scores = []
    for am in article_matrix:
        scores = torch.max(cosine_similarity(torch.tensor(am), torch.tensor(summary_matrix))).item()
        article_scores.append(scores)
    cosine.append(article_scores)

print(df_train.head())
print(cosine)
#time.sleep(1999)


# DATA, LABEL and SPLITTING
X = np.vstack(articles_train_tfidf)
y = np.array([score for article_scores in cosine for score in article_scores])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)





# WORD EMBEDDINGS



# SIMILARITY SCORING



# MODEL MAKING

# LINEAR REGRESSION
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_val)

# Perform cross-validation
cv_mse = cross_val_score(lr, X, y, scoring='neg_mean_squared_error', cv=5)
cv_r2 = cross_val_score(lr, X, y, scoring='r2', cv=5)

# Convert negative MSE to positive
cv_mse = -cv_mse

print(f"Cross-Validation MSE: {cv_mse.mean()} ± {cv_mse.std()}")
print(f"Cross-Validation R-squared: {cv_r2.mean()} ± {cv_r2.std()}")

mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"MSE {mse} \n MAE {mae} \n R2 {r2}")