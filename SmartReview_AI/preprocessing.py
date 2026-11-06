
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

contractions = {
    "can't": "cannot", "won't": "will not", "don't": "do not",
    "doesn't": "does not", "didn't": "did not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "wouldn't": "would not", "couldn't": "could not",
    "shouldn't": "should not", "i'm": "i am", "you're": "you are",
    "they're": "they are", "we're": "we are", "it's": "it is",
    "that's": "that is", "there's": "there is", "i've": "i have",
    "we've": "we have", "they've": "they have", "i'll": "i will",
    "you'll": "you will", "they'll": "they will"
}

stop_words = set(stopwords.words("english"))

# Preserve negation words for sentiment analysis
stop_words.difference_update(["never", "nor", "not", "no"])

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()

    for contraction, expansion in contractions.items():
        text = re.sub(
            r"\b" + re.escape(contraction) + r"\b",
            expansion,
            text
        )

    text = re.sub(r"\bw/\b", " with ", text)
    text = re.sub(r"\bu\b", " you ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)

    tokens = re.findall(r"[a-z]+", text)

    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(cleaned_tokens)
