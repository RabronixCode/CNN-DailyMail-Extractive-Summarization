
import time
from sklearn.ensemble import RandomForestRegressor
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
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import TrainingArguments, Trainer
from datasets import Dataset, DatasetDict, load_dataset
import evaluate

np.set_printoptions(threshold=np.inf)  # Print entire array without truncation
pd.set_option("display.width", 300)
"""
print(torch.cuda.is_available()) # Checking whether GPU is working

# Putting training data in df_train
df_train = pd.read_csv(r'C:\Users\User\Desktop\Python\CNN_DailyMail\Data\train.csv')

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
articles_train_tfidf, highlights_train_tfidf = tv.tfidf(articles_train, documents_articles, highlights_train, documents_highlights) # Returns transformed data

# SIMILARITY SCORING
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


# DATA, LABEL and SPLITTING
X = np.vstack(articles_train_tfidf)
y = np.array([score for article_scores in cosine for score in article_scores])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


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


# RANDOM FOREST
rf = RandomForestRegressor(n_estimators=100, oob_score=True)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_val)

# Perform cross-validation
cv_mse = cross_val_score(rf, X, y, scoring='neg_mean_squared_error', cv=5)
cv_r2 = cross_val_score(rf, X, y, scoring='r2', cv=5)

# Convert negative MSE to positive
cv_mse = -cv_mse

print(f"Cross-Validation MSE: {cv_mse.mean()} ± {cv_mse.std()}")
print(f"Cross-Validation R-squared: {cv_r2.mean()} ± {cv_r2.std()}")

mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
oob_score = rf.oob_score_

print(f"MSE {mse} \n MAE {mae} \n R2 {r2} \n OOB {oob_score}")
"""

# T5 (t5-large)
model = T5ForConditionalGeneration.from_pretrained("t5-large")
tokenizer = T5Tokenizer.from_pretrained("t5-large")

def preprocess_t5(examples):
    inputs = [f"summarize: {text}" for text in examples["article"]] # Makes a list of articles which are given prefix summarize so t5 knows what to do
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length") # Then we tokenize every input limited to 512

    labels = tokenizer(text_target=examples["highlights"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Load Train, Validation, and Test Datasets
dataset_files = {
    "train": r"C:\Users\User\Desktop\Python\CNN_DailyMail\Data\train.csv",  
    "validation": r"C:\Users\User\Desktop\Python\CNN_DailyMail\Data\validation.csv",  
    "test": r"C:\Users\User\Desktop\Python\CNN_DailyMail\Data\test.csv"  
}

# Load dataset directly from CSV files
dataset = DatasetDict({split: load_dataset("csv", data_files={split: path})[split].select(range(10)) for split, path in dataset_files.items()})

dataset = dataset.map(preprocess_t5, batched=True)

training_args = TrainingArguments(
    output_dir='./t5_results',  # Directory where model checkpoints & logs will be saved
    overwrite_output_dir=True,  # Overwrites old model checkpoints
    per_device_train_batch_size=1,  # Batch size for training per GPU/CPU
    per_device_eval_batch_size=1,  # Batch size for evaluation per GPU/CPU
    num_train_epochs=5,  # Number of epochs (full passes through the dataset)
    learning_rate=5e-5,  # Initial learning rate
    weight_decay=0.01,  # Regularization to prevent overfitting
    logging_dir='./t5-logs',  # Directory for logging
    logging_steps=100,  # Log loss after every X steps
    eval_strategy="epoch",  # Evaluate after every epoch ("steps" for more frequent evals)
    save_strategy="no",  # Save model after each epoch
    save_total_limit=2,  # Keep only the last 2 model checkpoints (delete older ones)
    report_to="tensorboard",  # Report logs to TensorBoard for visualization
    fp16=False,  # Use mixed-precision training (faster on GPUs)
    gradient_accumulation_steps=1,  # Accumulate gradients over multiple batches before updating weights
    warmup_steps=1000,  # Number of warmup steps for learning rate scheduler
    load_best_model_at_end=True,  # Load the best model (based on evaluation metric) after training
    metric_for_best_model="rougeL",  # Select best model based on ROUGE score
)

metric = evaluate.load("rouge")

def compute_metrics(eval_pred):
    """Compute ROUGE scores for summarization tasks."""
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {key: value.mid.fmeasure for key, value in result.items()}

trainer = Trainer(
    model=model,  # Your Transformer model (e.g., T5, BART)
    args=training_args,  # Training arguments (defined above)
    train_dataset=dataset["train"],  # Training dataset
    eval_dataset=dataset["validation"],  # Validation dataset (for evaluation)
    tokenizer=tokenizer,  # Tokenizer used for preprocessing
    compute_metrics=compute_metrics,  # Function to compute evaluation metrics (ROUGE, BLEU, etc.)
)

trainer.train()
metrics = trainer.evaluate()
print("Evaluation Metrics: ", metrics)