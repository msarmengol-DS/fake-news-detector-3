import streamlit as st
import string
import nltk
from gensim.models import Word2Vec
import numpy as np
import requests


nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
stop_words = set(stop_words)

EMBEDDING_DIM = 100
WORD2VEC_PATH = "models/word2vec.model"
API_URL = "http://127.0.0.1:5001/invocations"
HEADERS = {"Content-Type": "application/json"}

word2vec_model = Word2Vec.load(WORD2VEC_PATH)


def clean_word(word: str) -> str:
    """Remove punctuation and lowercase a word

    Args:
        word (str): the word to clean

    Returns:
        str: the cleaned word
    """
    word = word.lower()
    word = word.strip()

    for letter in word:
        if letter in string.punctuation:
            word = word.replace(letter, '')

    return word


def clean_text(text: str) -> list[str]:
    """Remove stop words and punctioation from a whole text.

    Args:
        text (str): the text to clean

    Returns:
        list[str]: the cleaned text
    """
    clean_text_list = []
    for word in text.split():
        cleaned_word = clean_word(word)
        if cleaned_word not in stop_words:
            clean_text_list.append(cleaned_word)

    return clean_text_list


def vectorize_text(text: list[str]) -> np.ndarray:
    """Vectorize a text by doing a sumatory of the word vectors

    Args:
        text (str): the text to vectorize

    Returns:
        np.ndarray: the vectorized text
    """
    text_vector = np.zeros(EMBEDDING_DIM, np.float32)
    for word in text:
        try:
            word_vector = word2vec_model.wv[word]
        except KeyError:
            st.warning(f"Word {word} not in vocabulary")
            continue
        text_vector += word_vector  # equivalent to text_vector = text_vector + word_vector

    return text_vector


def classify_embedding(embedding: np.ndarray) -> bool:
    """Classify a text by using ML model

    Args:
        embedding (np.ndarray): the vectorized text. Shape (100,)

    Returns:
        bool: True if the text is real, False otherwise
    """
    embedding = np.expand_dims(embedding, axis=0)  # changes from (100,) to (1, 100)

    data = {
        "inputs": embedding.tolist(),
    }

    response = requests.post(API_URL, json=data, headers=HEADERS)
    response_json = response.json()
    is_real = bool(response_json["predictions"][0])
    return is_real


st.title("Fake News Detector")
st.subheader("Detecting fake news with machine learning")

text_to_predict = st.text_area("Enter the new to check if is fake or not.")
button = st.button("Analyze")

if button:

    st.info("Cleaning text...")
    text_to_predict_clean = clean_text(text_to_predict)
    st.info("Vectorizing text...")
    text_to_predict_vectorized = vectorize_text(text_to_predict_clean)
    st.info("Classifying text...")
    is_real = classify_embedding(text_to_predict_vectorized)

    if is_real:
        st.success("Is real!")
    else:
        st.error("Is fake!")
