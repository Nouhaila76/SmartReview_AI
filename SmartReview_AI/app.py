
import streamlit as st
import joblib
import sys
import os

# Import preprocessing function
sys.path.append(os.path.dirname(__file__))
from preprocessing import preprocess_text

# Page configuration
st.set_page_config(
    page_title="SmartReview AI",
    page_icon="⭐",
    layout="wide"
)

# Load saved models
MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "smartreview_models"
)

@st.cache_resource
def load_models():
    sentiment_model = joblib.load(
        os.path.join(MODEL_DIR, "sentiment_model.pkl")
    )

    tfidf_vectorizer = joblib.load(
        os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    )

    return sentiment_model, tfidf_vectorizer

sentiment_model, tfidf_vectorizer = load_models()

# Application title
st.title("⭐ SmartReview AI")
st.subheader("Customer Sentiment and Topic Analysis")

st.write(
    "Analyze restaurant reviews using Natural Language Processing."
)

# Sidebar navigation
page = st.sidebar.selectbox(
    "Choose a feature",
    [
        "Sentiment Analysis",
        "Topic Analysis",
        "Suspicious Review Detection",
        "Dashboard"
    ]
)

# Sentiment Analysis
if page == "Sentiment Analysis":

    st.header("Customer Sentiment Analysis")

    review = st.text_area(
        "Enter a restaurant review:",
        placeholder="Example: The food was delicious and the service was excellent!"
    )

    if st.button("Analyze Sentiment"):

        if review.strip() == "":
            st.warning("Please enter a review.")

        else:
            cleaned_review = preprocess_text(review)

            if not cleaned_review.strip():
                st.warning("The review contains no usable words.")

            else:
                review_vector = tfidf_vectorizer.transform(
                    [cleaned_review]
                )

                prediction = sentiment_model.predict(
                    review_vector
                )[0]

                st.success(f"Predicted Sentiment: {prediction}")

                if hasattr(sentiment_model, "predict_proba"):
                    probabilities = sentiment_model.predict_proba(
                        review_vector
                    )[0]

                    st.subheader("Prediction Confidence")

                    for label, probability in zip(
                        sentiment_model.classes_,
                        probabilities
                    ):
                        st.write(
                            f"{label}: {probability:.2%}"
                        )

# Topic Analysis
elif page == "Topic Analysis":

    st.header("Restaurant Topic Analysis")

    topic_method = st.selectbox(
        "Choose a topic modeling method",
        ["LSA", "LDA", "BERTopic"]
    )

    review = st.text_area(
        "Enter a restaurant review:",
        placeholder="Example: The pizza was delicious and the service was excellent!"
    )

    if st.button("Analyze Topic"):

        if not review.strip():
            st.warning("Please enter a review.")

        else:
            cleaned_review = preprocess_text(review)

            if not cleaned_review.strip():
                st.warning("The review contains no usable words.")

            elif topic_method == "LSA":

                lsa_vectorizer = joblib.load(
                    os.path.join(MODEL_DIR, "lsa_vectorizer.pkl")
                )
                lsa_model = joblib.load(
                    os.path.join(MODEL_DIR, "lsa_model.pkl")
                )

                vector = lsa_vectorizer.transform([cleaned_review])
                topic_scores = lsa_model.transform(vector)[0]
                topic_id = topic_scores.argmax()

                st.success(f"Dominant LSA Topic: {topic_id + 1}")

            elif topic_method == "LDA":

                lda_vectorizer = joblib.load(
                    os.path.join(MODEL_DIR, "lda_vectorizer.pkl")
                )
                lda_model = joblib.load(
                    os.path.join(MODEL_DIR, "lda_model.pkl")
                )

                vector = lda_vectorizer.transform([cleaned_review])
                topic_scores = lda_model.transform(vector)[0]
                topic_id = topic_scores.argmax()

                st.success(f"Dominant LDA Topic: {topic_id + 1}")

            elif topic_method == "BERTopic":

                from bertopic import BERTopic

                @st.cache_resource
                def load_bertopic():
                    return BERTopic.load(
                        os.path.join(MODEL_DIR, "bertopic_model")
                    )

                topic_model = load_bertopic()
                topic_id, _ = topic_model.transform([cleaned_review])

                if topic_id[0] == -1:
                    st.warning("This review does not match a discovered topic.")
                else:
                    st.success(f"Detected BERTopic Topic: {topic_id[0]}")
                    topic_info = topic_model.get_topic(topic_id[0])
                    st.write("Main keywords:", ", ".join(
                        word for word, score in topic_info[:10]
                    ))

# Suspicious Review Detection
elif page == "Suspicious Review Detection":

    import re

    st.header("Suspicious Review Detection")

    st.write(
        "Identify reviews with potentially suspicious writing patterns."
    )

    review = st.text_area(
        "Enter a restaurant review:",
        placeholder="Paste a review here..."
    )

    if st.button("Check Review"):

        if not review.strip():
            st.warning("Please enter a review.")

        else:
            cleaned = preprocess_text(review)

            raw_words = re.findall(r"\b\w+\b", review)
            clean_words = cleaned.split()

            raw_word_count = len(raw_words)
            clean_word_count = len(clean_words)

            repetition_ratio = (
                1 - len(set(clean_words)) / len(clean_words)
                if clean_words else 0
            )

            generic_phrases = [
                "best ever", "must try", "highly recommend",
                "highly recommended", "worst ever", "never again",
                "life changing", "perfect place", "five stars",
                "best place in town"
            ]

            text_lower = review.lower()

            short_flag = clean_word_count < 5
            long_flag = raw_word_count > 472
            repetition_flag = (
                repetition_ratio > 0.40 and clean_word_count >= 10
            )
            generic_flag = any(
                phrase in text_lower for phrase in generic_phrases
            )

            score = sum([
                short_flag,
                long_flag,
                repetition_flag,
                generic_flag
            ])

            st.subheader("Review Analysis")

            st.write("Raw word count:", raw_word_count)
            st.write("Cleaned word count:", clean_word_count)
            st.write("Repetition ratio:", f"{repetition_ratio:.2%}")

            st.write("**Detected indicators:**")
            st.write("Very short review:", "Yes" if short_flag else "No")
            st.write("Unusually long review:", "Yes" if long_flag else "No")
            st.write("High repetition:", "Yes" if repetition_flag else "No")
            st.write("Generic phrases:", "Yes" if generic_flag else "No")

            st.metric("Suspicion Score", f"{score}/4")

            if score >= 2:
                st.warning(
                    "Potentially suspicious review: multiple indicators detected."
                )
            else:
                st.success(
                    "No sufficient suspicious indicators detected."
                )

# Dashboard
elif page == "Dashboard":

    import pandas as pd
    import matplotlib.pyplot as plt

    st.header("SmartReview AI Dashboard")

    st.write("Overview of the restaurant review dataset.")

    # Dataset statistics
    total_reviews = 14349
    positive_reviews = 9857
    negative_reviews = 2697
    neutral_reviews = 1795

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Reviews", f"{total_reviews:,}")
    col2.metric("Positive", f"{positive_reviews:,}")
    col3.metric("Negative", f"{negative_reviews:,}")
    col4.metric("Neutral", f"{neutral_reviews:,}")

    # Sentiment distribution
    st.subheader("Sentiment Distribution")

    sentiment_data = pd.DataFrame({
        "Sentiment": ["Positive", "Negative", "Neutral"],
        "Number of Reviews": [
            positive_reviews,
            negative_reviews,
            neutral_reviews
        ]
    })

    fig, ax = plt.subplots()

    ax.bar(
        sentiment_data["Sentiment"],
        sentiment_data["Number of Reviews"]
    )

    ax.set_ylabel("Number of Reviews")
    ax.set_title("Distribution of Customer Sentiments")

    st.pyplot(fig)
    plt.close(fig)

    # Project information
    st.subheader("Project Information")

    st.write("""
    **SmartReview AI** uses Natural Language Processing to:

    - Classify customer reviews into Positive, Neutral and Negative sentiments.
    - Discover hidden topics using LSA, LDA and BERTopic.
    - Identify potentially suspicious reviews using text-based heuristics.
    """)
