from datasets import load_dataset
from normalize_text_module import normalize_text
from gensim.models import KeyedVectors, Word2Vec
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

# Other pre-trained models

# Loading GloVe file
path = "./glove.6B/glove.6B.100d.txt"  # Checks if the glove file already exists in the system
if not os.path.exists(path):
    raise Exception("Glove file doesn't exist at the specified path in the code.")
glove_file = os.path.basename(path)  # File exists
print(f"Found glove file: {glove_file}")

output_file = (
    "glove.6B.100d.word2vec"  # The filename which will be stored as word2vec format
)
if not os.path.exists(output_file):
    print(f"Converting {os.path.basename(path)} to Word2Vec format...")
    glove2word2vec(
        glove_file, output_file
    )  # converts it into word2vec format and stores it
    pretrained_glove = KeyedVectors.load_word2vec_format(
        output_file, binary=False
    )  # loads the model

else:
    print(f"Loading existing {os.path.basename(output_file)}...")
    pretrained_glove = KeyedVectors.load_word2vec_format(
        output_file, binary=False
    )  # loads the model since it already exists

# Loading Word2Vec demo embeddings - Google News dataset
wv = api.load("word2vec-google-news-100")  # loads from gensim downloader


# The documentation said word2vec dataset has a drawback which doesn't infer vectors for unfamiliar words
# Created a function which checks if the word exists, otherwise raises an error
def get_key_from_w2v_demo(key):
    try:
        vector = wv[key]
        return vector
    except KeyError:
        print(f"The word {key} doesn't appear in this model")
