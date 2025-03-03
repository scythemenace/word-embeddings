from datasets import load_dataset
from gensim import models
from normalize_text_module import normalize_text
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
import os
import gensim.downloader as api

dataset = load_dataset("wikipedia", "20220301.simple")  # Loading the dataset

raw_text = dataset["train"][0]["text"]  # Extracting the dataset

# Processing the dataset using the normalize_text function from previous assignments
processed_text = normalize_text(
    raw_text,
    lemmatize=True,
    lowercase=True,
    remove_stopwords=True,
    remove_punctuation=True,
)


# Checks if a pre-trained model already exists and loads it, otherwise trains one from scratch and saves it
def get_model(filename, training_text, vector_size=100, window=5, min_count=1, sg=0):
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

vocab = model_sg.wv.index_to_key
print("Vocab of our trained models:\n", vocab)

# Other pre-trained models
# Loading GloVe file
pretrained_glove = api.load("glove-wiki-gigaword-100")

# Loading Word2Vec demo embeddings - Google News dataset
wv_demo = api.load("word2vec-google-news-300")  # loads from gensim downloader


# First Query
# print("Skip-gram results for 'female':")
# print(model_sg.wv.most_similar("female", topn=10))
# print("CBOW results for 'female':")
# print(model_cbow.wv.most_similar("female", topn=10))
# print("GloVe results for 'female':")
# print(pretrained_glove.most_similar("female", topn=10))
# print("Word2Vec Demo results for 'female':")
# print(wv_demo.most_similar("female", topn=10))
