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

# ADD CSS HERE
st.markdown("""
<style>

/* Main app background */
.stApp {
    background-color: #121212;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1E1E1E;
}

/* Main headings */
h1,h2,h3{
    color:#E50914;
}

/* Text */
p,label{
    color:white;
}

/* Inputs */
.stTextInput input,
.stNumberInput input{
    background:#2E2E2E;
    color:white;
}

/* Buttons */
.stButton>button{
    background:#E50914;
    color:white;
    border-radius:10px;
    height:3em;
    width:100%;
    font-size:18px;
}

.stButton>button:hover{
    background:#B20710;
}

/* Metrics */
div[data-testid="metric-container"]{
    background:#1E1E1E;
    border-radius:10px;
    padding:15px;
    border:1px solid #333;
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
The user selected the movie: {movie}

Recommended movie: {rec}

Explain why this recommendation is suitable.

Mention:

- Similar genres
- Similar storyline
- Similar audience preferences

Do NOT mention Content Score, Collaborative Score or Hybrid Score.

Write in simple English using 3–4 sentences.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# -------------------------------
# UI
# -------------------------------

# -------------------------------
# Sidebar
# -------------------------------

with st.sidebar:

    st.header("⚙ Recommendation Settings")
    
    movie = st.selectbox("🎬 Select a Movie",
        options=sorted(final_movies["title"].unique()))

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

tab1, tab2, tab3 = st.tabs(
    [
        "🎬 Recommendations",
        "📊 EDA Dashboard",
        "ℹ️ About Project"
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

            for _, row in result.iterrows():

                with st.container():

                    st.subheader(f"🎥 {row['title']}")
                    st.caption("Recommended especially for you")

                    st.progress(float(row["Hybrid Score"]))

                    st.success("### Why is this movie recommended?")

                    st.markdown("""
                    ✅ **Matches your favourite genres and storyline**

                    👥 **Users with similar preferences also watched and liked this movie**

                    ⭐ **Highly ranked by the Hybrid Recommendation Model**""")

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

    with tab2:

    st.title("📊 Exploratory Data Analysis")

    # -------------------------------------------------
    st.subheader("1️⃣ Distribution of Movie Ratings")

    st.image(
        "images/DistributionofMovieRatings.png",
        caption="Distribution of Movie Ratings",
        width="stretch"
    )

    st.info("""
    **Inference**

    • Most movies have ratings between **6 and 8**.

    • Very few movies receive extremely low ratings.

    • This indicates that the dataset mainly contains well-rated movies.
    """)

    st.divider()

    # -------------------------------------------------
    st.subheader("2️⃣ Top Genres")

    st.image(
        "images/top_genres.png",
        caption="Top Genres",
        width="stretch"
    )

    st.info("""
    **Inference**

    • Drama and Comedy are the most common genres.

    • Action and Adventure are also highly represented.

    • Genre information improves content-based recommendations.
    """)

    st.divider()

    # -------------------------------------------------
    st.subheader("3️⃣ Top Actors")

    st.image(
        "images/top_actor.png",
        caption="Top Actors",
        width="stretch"
    )

    st.info("""
    **Inference**

    • A few actors appear in many movies.

    • Popular actors contribute to movie similarity.

    • Frequent actor appearances improve recommendation quality.
    """)

    st.divider()

    # -------------------------------------------------
    st.subheader("4️⃣ Movies Released Per Year")

    st.image(
        "images/movie_released_peryear.png",
        caption="Movies Released Per Year",
        width="stretch"
    )

    st.info("""
    **Inference**

    • Movie production has increased steadily over the years.

    • Most movies in the dataset were released after 2000.

    • The dataset contains a strong representation of recent films.
    """)

    st.divider()

    # -------------------------------------------------
    st.subheader("5️⃣ Top Directors")

    st.image(
        "images/top_directors.png",
        caption="Top Directors",
        width="stretch"
    )

    st.info("""
    **Inference**

    • A few directors have contributed a large number of movies.

    • Director information helps identify similar movies.

    • Well-known directors dominate the dataset.
    """)

    st.divider()

    # -------------------------------------------------
    st.subheader("6️⃣ Top Most Popular Movies")

    st.image(
        "images/top_most_popular_movie.png",
        caption="Top Most Popular Movies",
        width="stretch"
    )

    st.info("""
    **Inference**

    • A small number of movies have very high popularity.

    • Popular movies are more likely to be recommended.

    • Popularity is an important feature in ranking recommendations.
    """)
        with tab3:

         st.title("🎬 About the Project")

         st.markdown("""
## AI Movie Recommendation System

This project recommends personalised movies using a Hybrid Recommendation System.

### Models Used

### 🎯 Content-Based Filtering

Uses **TF-IDF Vectorization** and **Cosine Similarity** to compare movie genres, keywords, cast, and descriptions. It recommends movies with similar content.

---

### 👥 Collaborative Filtering

Uses **Singular Value Decomposition (SVD)** from the Surprise library.

SVD is a **model-based collaborative filtering algorithm** that learns user preferences from historical ratings and predicts ratings for unseen movies.

---

### 🔀 Hybrid Recommendation

The final recommendation combines:

- 40% Content-Based Filtering
- 60% Collaborative Filtering (SVD)

This improves recommendation accuracy and personalization.

---

### 🤖 Explainable AI

Groq Llama 3.3 generates natural language explanations describing why each movie is recommended.
""")

### 📂 Dataset

- Movie Metadata
- User Ratings
- Movie Genres
- Movie Overview
- Popularity & Ratings

---

### ⚙️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- Surprise (SVD)
- Pandas
- Groq API
- Pickle

---

### 📈 Model Evaluation

st.subheader("📈 Model Performance")

metrics = pd.DataFrame({

"Model":[
"Content-Based",
"Collaborative (SVD)",
"Hybrid"
],

"Evaluation":[
"Precision@10",
"RMSE / MAE",
"Precision@10 & Recall@10"
],

"Result":[
"Your Precision",
"Your RMSE / MAE",
"Your Precision & Recall"
]

})

st.table(metrics)
### 👩‍💻 Developed By

**Prerana Gowda**

Business Analytics Student

AI & Machine Learning Enthusiast
""")
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
