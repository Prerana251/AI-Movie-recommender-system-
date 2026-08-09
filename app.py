import streamlit as st
import pickle
import pandas as pd
import os

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from groq import Groq


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="🎬 AI Movie Recommendation",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #121212;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #1E1E1E;
}

h1, h2, h3 {
    color: #E50914;
}

p, label {
    color: white;
}

.stTextInput input,
.stNumberInput input {
    background-color: #2E2E2E;
    color: white;
}

.stSelectbox div[data-baseweb="select"] > div {
    background-color: #2E2E2E;
    color: white;
}

.stButton > button {
    background-color: #E50914;
    color: white;
    border-radius: 10px;
    border: none;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

.stButton > button:hover {
    background-color: #B20710;
    color: white;
}

div[data-testid="metric-container"] {
    background-color: #1E1E1E;
    border-radius: 10px;
    padding: 15px;
    border: 1px solid #333;
}

.stProgress > div > div > div > div {
    background-color: #E50914;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.title("🎬 AI Movie Recommendation System")

st.markdown("""
Discover personalized movie recommendations using:

- 🎯 Content-Based Filtering
- 👥 Collaborative Filtering
- 🔀 Hybrid Recommendation
- 🤖 Explainable AI
""")


# =========================================================
# LOAD MODELS
# =========================================================

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


# =========================================================
# COSINE SIMILARITY
# =========================================================

tfidf_matrix = tfidf.transform(
    final_movies["tags"].fillna("")
)

cosine_sim = cosine_similarity(tfidf_matrix)


# =========================================================
# CONTENT-BASED RECOMMENDER
# =========================================================

def content_recommend(movie_title, top_n=100):

    matched = final_movies[
        final_movies["title"].str.lower()
        == movie_title.lower()
    ]

    if matched.empty:
        return pd.DataFrame(
            {"Message": ["Movie not found"]}
        )

    idx = matched.index[0]

    similarity_scores = list(
        enumerate(cosine_sim[idx])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )[1:top_n + 1]

    movie_indices = [
        i[0] for i in similarity_scores
    ]

    recommendations = final_movies.iloc[
        movie_indices
    ][
        ["movieId", "title"]
    ].copy()

    recommendations["Content Score"] = [
        i[1] for i in similarity_scores
    ]

    return recommendations.reset_index(drop=True)


# =========================================================
# COLLABORATIVE FILTERING - SVD
# =========================================================

def svd_recommend(user_id, top_n=None):

    watched = ratings_df[
        ratings_df["userId"] == user_id
    ]["movieId"].tolist()

    unwatched = final_movies[
        ~final_movies["movieId"].isin(watched)
    ].copy()

    unwatched["Collaborative Score"] = (
        unwatched["movieId"].apply(
            lambda x: svd_model.predict(
                user_id,
                x
            ).est
        )
    )

    recommendations = unwatched.sort_values(
        "Collaborative Score",
        ascending=False
    )

    if top_n:
        recommendations = recommendations.head(
            top_n
        )

    return recommendations[
        [
            "movieId",
            "title",
            "Collaborative Score"
        ]
    ]


# =========================================================
# HYBRID RECOMMENDER
# =========================================================

def hybrid_recommend(
    user_id,
    movie_title,
    top_n=10
):

    content = content_recommend(
        movie_title,
        100
    )

    if "Message" in content.columns:
        return content

    collaborative = svd_recommend(
        user_id
    )

    hybrid = pd.merge(
        content,
        collaborative,
        on=["movieId", "title"]
    )

    if hybrid.empty:

        return pd.DataFrame(
            {
                "Message": [
                    "No common recommendations found."
                ]
            }
        )

    content_scaler = MinMaxScaler()
    collaborative_scaler = MinMaxScaler()

    hybrid["Content Score"] = (
        content_scaler.fit_transform(
            hybrid[["Content Score"]]
        )
    )

    hybrid["Collaborative Score"] = (
        collaborative_scaler.fit_transform(
            hybrid[["Collaborative Score"]]
        )
    )

    # 40% Content + 60% Collaborative
    hybrid["Hybrid Score"] = (
        0.4 * hybrid["Content Score"]
        +
        0.6 * hybrid["Collaborative Score"]
    )

    hybrid = hybrid.sort_values(
        "Hybrid Score",
        ascending=False
    )

    return hybrid.head(top_n)


# =========================================================
# COLD START
# =========================================================

def is_new_user(user_id):

    return user_id not in (
        ratings_df["userId"].unique()
    )


def cold_start_recommend(
    movie_title,
    top_n=10
):

    recommendations = content_recommend(
        movie_title,
        top_n=100
    )

    if "Message" in recommendations.columns:
        return recommendations

    recommendations[
        "Recommendation Type"
    ] = "New User - Content Based"

    return recommendations.head(
        top_n
    )


def recommend_with_cold_start(
    user_id,
    movie_title,
    top_n=10
):

    if is_new_user(user_id):

        return cold_start_recommend(
            movie_title,
            top_n
        )

    return hybrid_recommend(
        user_id,
        movie_title,
        top_n
    )


# =========================================================
# GROQ EXPLAINABLE AI
# =========================================================

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)


@st.cache_data
def explain(movie, rec):

    selected_row = final_movies[
        final_movies["title"].str.lower()
        == movie.lower()
    ]

    recommended_row = final_movies[
        final_movies["title"].str.lower()
        == rec.lower()
    ]

    if (
        selected_row.empty
        or recommended_row.empty
    ):

        return (
            "Recommended because it matches "
            "your movie preferences."
        )

    selected_tags = str(
        selected_row.iloc[0]["tags"]
    )

    recommended_tags = str(
        recommended_row.iloc[0]["tags"]
    )

    prompt = f"""
The user selected this movie:

{movie}

Selected movie characteristics:
{selected_tags}

Recommended movie:
{rec}

Recommended movie characteristics:
{recommended_tags}

Give ONLY ONE short reason.

Maximum 15 words.

Start with:
"Recommended because..."

Mention only relevant factors such as:
- similar genres
- similar themes
- similar storyline
- similar audience preferences

Do not mention:
- movie names
- AI
- algorithms
- Content Score
- Collaborative Score
- Hybrid Score
- numbers
- scores
- ratings

Do not invent information.

Example:
Recommended because it shares similar genres and themes with your selected movie.
"""

    try:

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=40
        )

        return (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

    except Exception:

        return (
            "Recommended because it shares "
            "similar genres and themes."
        )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Recommendation Settings")

    movie = st.selectbox(
        "🎬 Select a Movie",
        options=sorted(
            final_movies["title"]
            .dropna()
            .unique()
        )
    )

    user = st.number_input(
        "👤 User ID",
        min_value=1,
        value=1,
        step=1
    )

    top = st.slider(
        "🎯 Number of Recommendations",
        5,
        20,
        10
    )

    recommend = st.button(
        "🎬 Recommend"
    )


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🎬 Recommendations",
        "📊 EDA Dashboard",
        "ℹ️ About Project"
    ]
)


# =========================================================
# RECOMMENDATION PAGE
# =========================================================

with tab1:

    if recommend:

        if not movie:

            st.warning(
                "Please select a movie."
            )

            st.stop()

        with st.spinner(
            "🔍 Finding the best movies for you..."
        ):

            result = recommend_with_cold_start(
                user_id=user,
                movie_title=movie,
                top_n=top
            )

        # ---------------------------------------------
        # ERROR
        # ---------------------------------------------

        if "Message" in result.columns:

            st.error(
                result["Message"].iloc[0]
            )

        else:

            # -----------------------------------------
            # USER TYPE
            # -----------------------------------------

            if is_new_user(user):

                st.info(
                    "🆕 New user detected. "
                    "Recommendations are based on "
                    "your selected movie preferences."
                )

            else:

                st.success(
                    "👤 Existing user detected. "
                    "Recommendations are personalized "
                    "using your previous rating preferences "
                    "and movie similarity."
                )

            # -----------------------------------------
            # METRICS
            # -----------------------------------------

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "🎬 Movies Recommended",
                len(result)
            )

            col2.metric(
                "🎯 Selected Movie",
                movie
            )

            col3.metric(
                "⭐ Top Recommendation",
                result.iloc[0]["title"]
            )

            st.markdown("---")

            # -----------------------------------------
            # RECOMMENDED MOVIES
            # -----------------------------------------

            st.header(
                "🎬 Recommended Movies"
            )

            st.caption(
                "Recommendations are based on your "
                "movie preferences, previous viewing/rating "
                "history, and similar movie characteristics."
            )

            for i, (_, row) in enumerate(
                result.iterrows(),
                start=1
            ):

                # Movie title only
                st.subheader(
                    f"{i}. 🎬 {row['title']}"
                )

                # -------------------------------------
                # ONE SHORT GROQ REASON
                # -------------------------------------

                explanation = explain(
                    movie,
                    row["title"]
                )

                st.info(
                    f"🎯 {explanation}"
                )

                st.divider()


# =========================================================
# EDA DASHBOARD
# =========================================================

def show_image(
    primary_path,
    fallback_path,
    caption
):

    if os.path.exists(primary_path):

        image_path = primary_path

    elif os.path.exists(fallback_path):

        image_path = fallback_path

    else:

        st.warning(
            f"EDA image not found: {primary_path}"
        )

        return

    # Center the image
    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:

        st.image(
            image_path,
            caption=caption,
            width=600
        )


with tab2:

    st.title(
        "📊 Exploratory Data Analysis"
    )

    st.markdown(
        "### Dataset Insights"
    )


    # =====================================================
    # 1. RATINGS
    # =====================================================

    st.subheader(
        "1️⃣ Distribution of Movie Ratings"
    )

    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:

        st.image(
            "images/DistributionofMovieRatings.png",
            caption="Distribution of Movie Ratings",
            width=600
        )

    st.info("""
    **Inference**

    • Most movie ratings are concentrated around the middle-to-high range.

    • Very low and extremely high ratings occur less frequently.

    • This indicates that the dataset contains a large proportion of moderately to highly rated movies.
    """)

    st.divider()


    # =====================================================
    # 2. GENRES
    # =====================================================

    st.subheader(
        "2️⃣ Top Genres"
    )

    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:

        st.image(
            "images/top_genres.png",
            caption="Top Genres",
            width=600
        )

    st.info("""
    **Inference**

    • The dataset contains a strong representation of popular movie genres.

    • Drama, Comedy, Action and Adventure are among the commonly represented genres.

    • Genre information is useful for content-based movie similarity.
    """)

    st.divider()


    # =====================================================
    # 3. ACTORS
    # =====================================================

    st.subheader(
        "3️⃣ Top Actors"
    )

    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:

        st.image(
            "images/top_actor.png",
            caption="Top Actors",
            width=600
        )

    st.info("""
    **Inference**

    • Some actors appear more frequently than others in the dataset.

    • Actor information can help identify movies with similar cast characteristics.

    • Cast-related features can therefore contribute to content-based recommendations.
    """)

    st.divider()


    # =====================================================
    # 4. MOVIES PER YEAR
    # =====================================================

    st.subheader(
        "4️⃣ Movies Released Per Year"
    )

    show_image(
        "images/movie_released_peryear.png",
        "images/movie_relased_peryear.png",
        "Movies Released Per Year"
    )

    st.info("""
    **Inference**

    • The number of movies varies across different release years.

    • The dataset contains substantial representation from modern movie releases.

    • Release-year information can help understand the time distribution of the movie catalogue.
    """)

    st.divider()


    # =====================================================
    # 5. DIRECTORS
    # =====================================================

    st.subheader(
        "5️⃣ Top Directors"
    )

    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:

        st.image(
            "images/top_directors.png",
            caption="Top Directors",
            width=600
        )

    st.info("""
    **Inference**

    • A small group of directors appears more frequently in the dataset.

    • Director information can help identify movies with similar creative characteristics.

    • This feature can contribute to content-based recommendation.
    """)

    st.divider()


    # =====================================================
    # 6. POPULAR MOVIES
    # =====================================================

    st.subheader(
        "6️⃣ Top Most Popular Movies"
    )

    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:

        st.image(
            "images/top_most_popular_movie.png",
            caption="Top Most Popular Movies",
            width=600
        )

    st.info("""
    **Inference**

    • A relatively small group of movies has substantially higher popularity.

    • Popularity can be useful as an additional signal when analysing movie demand.

    • However, the recommendation system primarily relies on user preferences and movie similarity.
    """)


# =========================================================
# ABOUT PROJECT
# =========================================================

with tab3:

    st.title(
        "🎬 About the Project"
    )

    st.markdown("""
## AI Movie Recommendation System

This project recommends personalized movies using a Hybrid Recommendation System.

### 📂 Dataset

- Movie Metadata
- User Ratings
- Movie Genres
- Movie Overview
- Popularity
- Movie Ratings

---

### ⚙️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- Surprise
- SVD
- Pandas
- Groq API

---

## 🧠 Models Used

### 🎯 Content-Based Filtering

Uses **TF-IDF Vectorization** and **Cosine Similarity** to compare movie characteristics such as genres, keywords, cast and descriptions.

The model recommends movies that are similar to the movie selected by the user.

---

### 👥 Collaborative Filtering

Uses **Singular Value Decomposition (SVD)** from the Surprise library.

SVD learns user preferences from historical ratings and predicts preferences for movies that the user has not previously rated.

---

### 🔀 Hybrid Recommendation

The final recommendation combines:

- **40% Content-Based Filtering**
- **60% Collaborative Filtering**

This allows the system to combine movie similarity with learned user preferences.

---

### 🆕 Cold Start Recommendation

For a new user who has no previous rating history, collaborative filtering cannot learn personal preferences.

Therefore, the system uses **Content-Based Filtering** and recommends movies similar to the movie initially selected by the new user.

---

### 🤖 Explainable AI

Groq Llama 3.3 generates a short, plain-English reason explaining why each recommended movie was selected.
""")

    # =====================================================
    # MODEL PERFORMANCE
    # =====================================================

    st.subheader(
        "📈 Model Performance"
    )

    metrics = pd.DataFrame({
        "Model": [
            "Collaborative Filtering (SVD)",
            "Hybrid Recommendation"
        ],

        "Evaluation": [
            "RMSE / MAE",
            "Precision@10 & Recall@10"
        ],

        "Result": [
            "RMSE: 0.88 | MAE: 0.68",
            "Precision@10: 1.00 | Recall@10: 0.0004"
        ]
    })

    st.table(metrics)

    st.markdown("""
### 📌 Interpretation

**SVD:**  
RMSE and MAE measure how accurately the collaborative filtering model predicts user ratings.

**Hybrid Model:**  
Precision@10 measures how many of the recommended movies are relevant, while Recall@10 measures how many relevant movies were successfully retrieved within the top 10 recommendations.
""")


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
### 👩‍💻 Developed by

**Prerana C **

---

### 🎬 AI Movie Recommendation System

✔ Content-Based Filtering

✔ Collaborative Filtering (SVD)

✔ Hybrid Recommendation

✔ Cold Start Handling

✔ Explainable AI
""")
