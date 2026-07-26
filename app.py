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
    page_title="🎬 AI Movie Recommendation",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>

.main{
    background-color:#fafafa;
}

.stButton>button{
    background-color:#E50914;
    color:white;
    border-radius:10px;
    height:3em;
    width:100%;
    font-size:18px;
}

.stButton>button:hover{
    background-color:#b20710;
    color:white;
}

div[data-testid="metric-container"]{
    background:#f8f9fa;
    padding:15px;
    border-radius:10px;
    border:1px solid #ddd;
}

h1{
    color:#E50914;
}

</style>
""", unsafe_allow_html=True)

st.title("🎬 AI Movie Recommendation System")

st.markdown("""
Discover personalized movie recommendations using

- 🎯 Content-Based Filtering
- 👥 Collaborative Filtering
- 🤖 Explainable AI (Groq)
""")

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
    api_key=st.secrets["GROQ_API_KEY"]
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

# Sidebar
with st.sidebar:

    st.header("⚙ Recommendation Settings")

    movie = st.text_input(
        "🎬 Movie Title",
        placeholder="Titanic"
    )

    user = st.number_input(
        "👤 User ID",
        min_value=1,
        value=1
    )

    top = st.slider(
        "🎯 Number of Recommendations",
        5,
        20,
        10
    )

    recommend = st.button("🎬 Recommend")
    # -------------------------------
# Tabs
# -------------------------------

tab1, tab2 = st.tabs(
    [
        "🎬 Recommendations",
        "📊 EDA Dashboard"
    ]
)


# Main Page
with tab1:

    if recommend:

    if movie.strip() == "":
        st.warning("Please enter a movie title.")
        st.stop()

    with st.spinner("🔍 Finding the best movies for you..."):

        result = hybrid_recommend(user, movie, top)

    if "Message" in result.columns:

        st.error("Movie not found.")

    else:

        st.success("Recommendations generated successfully!")

        # -------------------------
        # Metrics
        # -------------------------

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Movies Found",
            len(result)
        )

        col2.metric(
            "Average Score",
            round(result["Hybrid Score"].mean(), 2)
        )

        col3.metric(
            "Best Match",
            result.iloc[0]["title"]
        )

        st.markdown("---")

        st.header("🎬 Recommended Movies")

        # -------------------------
        # Movie Cards
        # -------------------------

        for _, row in result.iterrows():

            with st.container():

                st.subheader(f"🎥 {row['title']}")

                st.progress(float(row["Hybrid Score"]))

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Content",
                    f"{row['Content Score']:.2f}"
                )

                c2.metric(
                    "Collaborative",
                    f"{row['Collaborative Score']:.2f}"
                )

                c3.metric(
                    "Hybrid",
                    f"{row['Hybrid Score']:.2f}"
                )

                with st.expander("🤖 Why did AI recommend this movie?"):

                    explanation = explain(
                        movie,
                        row["title"],
                        row["Content Score"],
                        row["Collaborative Score"],
                        row["Hybrid Score"]
                    )

                    st.write(explanation)

                st.markdown("---")
# -------------------------------
# EDA Dashboard
# -------------------------------

with tab2:

    st.title("📊 Exploratory Data Analysis")

    st.markdown("### Dataset Insights")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            "images/DistributionofMovieRatings.png",
            caption="Distribution of Movie Ratings",
            use_container_width=True
        )

        st.image(
            "images/top_genres.png",
            caption="Top Genres",
            use_container_width=True
        )

        st.image(
            "images/top_actor.png",
            caption="Top Actors",
            use_container_width=True
        )

        st.image(
            "images/movie_released_peryear.png",
            caption="Movies Released Per Year",
            use_container_width=True
        )

    with col2:

        st.image(
            "images/popularity_distribution.png",
            caption="Popularity Distribution",
            use_container_width=True
        )

        st.image(
            "images/top_directors.png",
            caption="Top Directors",
            use_container_width=True
        )

        st.image(
            "images/top_most_popular_movie.png",
            caption="Top Most Popular Movies",
            use_container_width=True
        )

# Footer

st.markdown("---")

st.markdown(
"""
### 👩‍💻 Developed by

**Prerana Gowda**

🎬 AI Movie Recommendation System

✔ Content-Based Filtering

✔ Collaborative Filtering (SVD)

✔ Hybrid Recommendation

✔ Explainable AI (Groq)
"""
)