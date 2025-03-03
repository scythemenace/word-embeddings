from datasets import load_dataset
from nltk.stem.snowball import stopwords
from normalize_text_module import normalize_text
from gensim.models import Word2Vec
import os

dataset = load_dataset("wikipedia", "20220301.simple")

raw_text = dataset["train"][0]["text"]

processed_text = normalize_text(
    raw_text,
    lemmatize=True,
    lowercase=True,
    remove_stopwords=True,
    remove_punctuation=True,
)


def get_model(filename, training_text, vector_size=100, window=5, min_count=1, sg=0):
    if os.path.exists(filename):
        print(f"Loading existing module from {filename}...")
        return Word2Vec.load(filename)

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


model_sg = get_model("skipgram_model.model", processed_text)
print(model_sg.wv["april"])
