from datasets import load_dataset
from gensim import models
from normalize_text_sentence_module import normalize_text_sentence
from normalize_text_module import normalize_text
from gensim.models import Word2Vec
import os
import gensim.downloader as api
from wefe.metrics import WEAT
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

dataset = load_dataset("wikipedia", "20220301.simple")  # Loading the dataset

raw_text = dataset["train"][0]["text"]  # Extracting the dataset

# Processing the dataset using the normalize_text function from previous assignments
processed_text = normalize_text_sentence(raw_text, remove_stopwords=True, lowercase=True, remove_punctuation=True)


# Checks if a pre-trained model already exists and loads it, otherwise trains one from scratch and saves it
def get_model(filename, training_text, vector_size=100, window=10, min_count=1, sg=0):
    # Checks if the file exists
    if os.path.exists(filename):
        print(f"Loading existing module from {filename}...")
        return Word2Vec.load(filename)

    # Load from scatch
    else:
        print("Model doesn't exist in the system...")
        print("Training model from scratch...")
        model = Word2Vec(
            training_text,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg,
        )

        model.save(filename)
        return model


# My models (Skip-gram and CBOW)
model_sg = get_model("skipgram_model.model", processed_text, sg=1)
model_cbow = get_model("cbow_model.model", processed_text)

# vocab = model_sg.wv.index_to_key
# print("Vocab of our trained models:\n", vocab)

# Other pre-trained models
# Loading GloVe file
pretrained_glove = api.load("glove-wiki-gigaword-100")

# Loading Word2Vec demo embeddings - Google News dataset
wv_demo = api.load("word2vec-google-news-300")  # loads from gensim downloader


# First Query - "female"
print("Skip-gram results for 'female':")
print(model_sg.wv.most_similar("female", topn=10))
print("CBOW results for 'female':")
print(model_cbow.wv.most_similar("female", topn=10))
print("GloVe results for 'female':")
print(pretrained_glove.most_similar("female", topn=10))
print("Word2Vec Demo results for 'female':")
print(wv_demo.most_similar("female", topn=10))

# Second Query - "united - america + england"
print("Skip-gram results for 'united - america + england':")
print(
    model_sg.wv.most_similar(
        positive=["united", "england"], negative=["america"], topn=10
    )
)

print("CBOW results for 'united - america + england':")
print(
    model_cbow.wv.most_similar(
        positive=["united", "england"], negative=["america"], topn=10
    )
)

print("GloVe results for 'united - america + england':")
print(
    pretrained_glove.most_similar(
        positive=["united", "england"], negative=["america"], topn=10
    )
)

print("Word2Vec Demo results for 'united - america + england':")
print(
    wv_demo.most_similar(positive=["united", "england"], negative=["america"], topn=10)
)

# Third Query - "day"
print("Skip-gram results for 'day':")
print(model_sg.wv.most_similar("day", topn=10))
print("CBOW results for 'day':")
print(model_cbow.wv.most_similar("day", topn=10))
print("GloVe results for 'day':")
print(pretrained_glove.most_similar("day", topn=10))
print("Word2Vec Demo results for 'day':")
print(wv_demo.most_similar("day", topn=10))

# Fourth Query - "world - war + peace"
print("Skip-gram results for 'world - war + peace':")
print(model_sg.wv.most_similar(positive=["world", "peace"], negative=["war"], topn=10))

print("CBOW results for 'world - war + peace':")
print(
    model_cbow.wv.most_similar(positive=["world", "peace"], negative=["war"], topn=10)
)

print("GloVe results for 'world - war + peace':")
print(
    pretrained_glove.most_similar(
        positive=["world", "peace"], negative=["war"], topn=10
    )
)

print("Word2Vec Demo results for 'world - war + peace':")
print(wv_demo.most_similar(positive=["world", "peace"], negative=["war"], topn=10))

# Fifth Query - "year"
print("Skip-gram results for 'year':")
print(model_sg.wv.most_similar("year", topn=10))
print("CBOW results for 'year':")
print(model_cbow.wv.most_similar("year", topn=10))
print("GloVe results for 'year':")
print(pretrained_glove.most_similar("year", topn=10))
print("Word2Vec Demo results for 'year':")
print(wv_demo.most_similar("year", topn=10))

# Conducting WEAT test similar to examples provided in the assignment

# First concept pair: Time periods (Past vs. Future)
past_terms = ["history", "old", "previous", "ancient", "traditional"]
future_terms = ["new", "become", "change", "create", "development"]

# Second concept pair: Celebration vs. Conflict
celebration_terms = ["festival", "celebrate", "holiday", "birthday", "party"]
conflict_terms = ["war", "force", "battle", "fight", "conflict"]

# Check vocabulary coverage
print("\nChecking vocabulary coverage...")
all_weat_terms = past_terms + future_terms + celebration_terms + conflict_terms


# Filter terms based on vocabulary presence because otherwise we're getting warnings that the word doesn't exist in our model's vocab
def filter_terms(terms, model):
    filtered = [term for term in terms if term in model.key_to_index]
    return filtered


# Check and filter for Skip-gram model
print("Skip-gram model:")
past_terms_sg = filter_terms(past_terms, model_sg.wv)
future_terms_sg = filter_terms(future_terms, model_sg.wv)
celebration_terms_sg = filter_terms(celebration_terms, model_sg.wv)
conflict_terms_sg = filter_terms(conflict_terms, model_sg.wv)

# Check and filter for CBOW model
print("CBOW model:")
male_terms_cbow = filter_terms(past_terms, model_cbow.wv)
female_terms_cbow = filter_terms(future_terms, model_cbow.wv)
career_terms_cbow = filter_terms(celebration_terms, model_cbow.wv)
family_terms_cbow = filter_terms(conflict_terms, model_cbow.wv)


target_sets = ["past_terms", "future_terms"]
attribute_sets = ["career_terms", "family_terms"]

# Create a query for each model
sg_query = Query(
    [past_terms_sg, future_terms_sg],
    [celebration_terms_sg, conflict_terms_sg],
    target_sets,
    attribute_sets,
)

cbow_query = Query(
    [male_terms_cbow, female_terms_cbow],
    [career_terms_cbow, family_terms_cbow],
    target_sets,
    attribute_sets,
)

# Original query for pre-trained models
original_query = Query(
    [past_terms, future_terms],
    [celebration_terms, conflict_terms],
    target_sets,
    attribute_sets,
)

# Embedding models
embedded_w2v = WordEmbeddingModel(wv_demo, "GloVe Model")
embedded_glove = WordEmbeddingModel(pretrained_glove, "word2vec Model")

# Create WEFE model wrappers
wefe_sg = WordEmbeddingModel(model_sg.wv, "Skip-gram Model")
wefe_cbow = WordEmbeddingModel(model_cbow.wv, "CBOW Model")
wefe_glove = WordEmbeddingModel(pretrained_glove, "GloVe Model")
wefe_w2v = WordEmbeddingModel(wv_demo, "Word2Vec Google News")

# Run WEAT tests
weat_metric = WEAT()

print("\nRunning WEAT tests...")

# Test Skip-gram
print("\nSkip-gram Model:")
if all(
    [
        len(past_terms_sg) > 0,
        len(future_terms_sg) > 0,
        len(celebration_terms_sg) > 0,
        len(conflict_terms_sg) > 0,
    ]
):
    wefe_result = weat_metric.run_query(sg_query, wefe_sg)
    print(wefe_result)
else:
    print("Not enough terms in vocabulary for WEAT test")

# Test CBOW
print("\nCBOW Model:")
if all(
    [
        len(male_terms_cbow) > 0,
        len(female_terms_cbow) > 0,
        len(career_terms_cbow) > 0,
        len(family_terms_cbow) > 0,
    ]
):
    wefe_result = weat_metric.run_query(cbow_query, wefe_cbow)
    print(wefe_result)
else:
    print("Not enough terms in vocabulary for WEAT test")

# Test pre-trained models
print("\nGloVe Model:")
wefe_result = weat_metric.run_query(original_query, wefe_glove)
print(wefe_result)

print("\nWord2Vec Google News Model:")
wefe_result = weat_metric.run_query(original_query, wefe_w2v)
print(wefe_result)


# Load a text classification dataset (20 Newsgroups subset)
categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

# Normalize and preprocess text
X_train = [" ".join(normalize_text(text, lowercase=True)) for text in newsgroups_train.data]
X_test = [" ".join(normalize_text(text, lowercase=True)) for text in newsgroups_test.data]
y_train = newsgroups_train.target
y_test = newsgroups_test.target

# 1. Bag-of-Words Model
bow_vectorizer = TfidfVectorizer(max_features=5000)
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

# Train and evaluate BoW model
bow_model = LogisticRegression(max_iter=1000)
bow_model.fit(X_train_bow, y_train)
bow_pred = bow_model.predict(X_test_bow)

bow_acc = accuracy_score(y_test, bow_pred)
bow_f1 = f1_score(y_test, bow_pred)
print(f"Bag-of-Words Results - Accuracy: {bow_acc:.4f}, F1: {bow_f1:.4f}")

# 2. Averaged Word Embeddings Model (Using pretrained GloVe)
def document_to_vector(text, model):
    words = text.split()
    vectors = [model[word] for word in words if word in model.key_to_index]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

X_train_emb = np.array([document_to_vector(text, pretrained_glove) for text in X_train])
X_test_emb = np.array([document_to_vector(text, pretrained_glove) for text in X_test])

# Train and evaluate embeddings model
emb_model = LogisticRegression(max_iter=1000)
emb_model.fit(X_train_emb, y_train)
emb_pred = emb_model.predict(X_test_emb)

emb_acc = accuracy_score(y_test, emb_pred)
emb_f1 = f1_score(y_test, emb_pred)
print(f"Embedding Results - Accuracy: {emb_acc:.4f}, F1: {emb_f1:.4f}")

# Feature dimension comparison
print(f"\nFeature Dimensions:\nBoW: {X_train_bow.shape[1]}\nEmbeddings: {pretrained_glove.vector_size}")