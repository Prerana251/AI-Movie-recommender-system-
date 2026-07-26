import streamlit as st
import pickle
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from groq import Groq

# -------------------------------
# Page Config
# -------------------------------

st.set_page_config(
    page_title="AI Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 AI Movie Recommendation System")

# -------------------------------
# Load Models
# -------------------------------

@st.cache_resource
def load_models():

    with open("models/tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)

    with open("models/final_movies.pkl", "rb") as f:
        final_movies = pickle.load(f)

    with open("models/ratings.pkl", "rb") as f:
        ratings_df = pickle.load(f)

    with open("models/svd_model.pkl", "rb") as f:
        svd_model = pickle.load(f)

    return tfidf, final_movies, ratings_df, svd_model


tfidf, final_movies, ratings_df, svd_model = load_models()

# Build cosine similarity (instead of loading the 627 MB file)
tfidf_matrix = tfidf.transform(final_movies["tags"])
cosine_sim = cosine_similarity(tfidf_matrix)

# -------------------------------
# Recommendation Functions
# -------------------------------

def content_recommend(movie_title, top_n=100):

    matched = final_movies[
        final_movies["title"].str.lower() == movie_title.lower()
    ]

    if matched.empty:
        return pd.DataFrame({"Message": ["Movie not found"]})

    idx = matched.index[0]

    similarity_scores = list(enumerate(cosine_sim[idx]))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )[1:top_n+1]

    movie_indices = [i[0] for i in similarity_scores]

    recommendations = final_movies.iloc[movie_indices][
        ["movieId", "title"]
    ].copy()

    recommendations["Content Score"] = [
        i[1] for i in similarity_scores
    ]

    return recommendations.reset_index(drop=True)


def svd_recommend(user_id, top_n=None):

    watched = ratings_df[
        ratings_df["userId"] == user_id
    ]["movieId"].tolist()

    unwatched = final_movies[
        ~final_movies["movieId"].isin(watched)
    ].copy()

    unwatched["Collaborative Score"] = unwatched["movieId"].apply(
        lambda x: svd_model.predict(user_id, x).est
    )

    recommendations = unwatched.sort_values(
        "Collaborative Score",
        ascending=False
    )

    if top_n:
        recommendations = recommendations.head(top_n)

    return recommendations[
        [
            "movieId",
            "title",
            "Collaborative Score"
        ]
    ]


def hybrid_recommend(user_id, movie_title, top_n=10):

    content = content_recommend(movie_title, 100)

    if "Message" in content.columns:
        return content

    collaborative = svd_recommend(user_id)

    hybrid = pd.merge(
        content,
        collaborative,
        on=["movieId", "title"]
    )

    scaler = MinMaxScaler()

    hybrid["Content Score"] = scaler.fit_transform(
        hybrid[["Content Score"]]
    )

    hybrid["Collaborative Score"] = scaler.fit_transform(
        hybrid[["Collaborative Score"]]
    )

    hybrid["Hybrid Score"] = (
        0.4 * hybrid["Content Score"] +
        0.6 * hybrid["Collaborative Score"]
    )

    hybrid = hybrid.sort_values(
        "Hybrid Score",
        ascending=False
    )

    return hybrid.head(top_n)

# -------------------------------
# Groq
# -------------------------------

client = Groq(
    api_key="GROQ_API_KEY"
)


def explain(movie, rec, c, cf, h):

    prompt = f"""
The user searched for {movie}.

Recommended movie:
{rec}

Content Score: {c:.2f}
Collaborative Score: {cf:.2f}
Hybrid Score: {h:.2f}

Explain in plain English in 3 sentences.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# -------------------------------
# UI
# -------------------------------

movie = st.text_input("Movie Title")

user = st.number_input(
    "User ID",
    min_value=1,
    step=1
)

top = st.slider(
    "Recommendations",
    5,
    20,
    10
)

if st.button("Recommend"):

    result = hybrid_recommend(user, movie, top)

    if "Message" in result.columns:

        st.error("Movie not found.")

    else:

        st.subheader("Recommended Movies")

        st.dataframe(
            result[
                [
                    "title",
                    "Content Score",
                    "Collaborative Score",
                    "Hybrid Score"
                ]
            ]
        )

        st.subheader("AI Explanations")

        for _, row in result.iterrows():

            with st.expander(row["title"]):

                st.write(
                    explain(
                        movie,
                        row["title"],
                        row["Content Score"],
                        row["Collaborative Score"],
                        row["Hybrid Score"]
                    )
                )