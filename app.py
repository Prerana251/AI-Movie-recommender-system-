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

# -------------------------------
# CSS
# -------------------------------

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
h1, h2, h3 {
    color: #E50914;
}

/* Text */
p, label {
    color: white;
}

/* Inputs */
.stTextInput input,
.stNumberInput input {
    background: #2E2E2E;
    color: white;
}

/* Buttons */
.stButton > button {
    background: #E50914;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

.stButton > button:hover {
    background: #B20710;
}

/* Metrics */
div[data-testid="metric-container"] {
    background: #1E1E1E;
    border-radius: 10px;
    padding: 15px;
    border: 1px solid #333;
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

# Build cosine similarity
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
    )[1:top_n + 1]

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
        0.4 * hybrid["Content Score"]
        + 0.6 * hybrid["Collaborative Score"]
    )

    hybrid = hybrid.sort_values(
        "Hybrid Score",
        ascending=False
    )

    return hybrid.head(top_n)

# -------------------------------
# Cold Start Handling
# -------------------------------

def is_new_user(user_id):
    """
    Check whether the user has any historical ratings.
    """
    return user_id not in ratings_df["userId"].unique()


def cold_start_recommend(movie_title, top_n=10):
    """
    Recommendation strategy for a completely new user.

    Since a new user has no rating history, SVD cannot
    personalize recommendations. We therefore use
    content-based filtering based on the movie selected
    by the user.
    """

    recommendations = content_recommend(
        movie_title,
        top_n=100
    )

    if "Message" in recommendations.columns:
        return recommendations

    recommendations["Recommendation Type"] = (
        "New User - Based on your selected movie"
    )

    return recommendations.head(top_n)


def recommend_with_cold_start(user_id, movie_title, top_n=10):
    """
    Route new users to cold-start recommendations
    and existing users to the hybrid recommender.
    """

    if is_new_user(user_id):

        return cold_start_recommend(
            movie_title,
            top_n
        )

    else:

        return hybrid_recommend(
            user_id,
            movie_title,
            top_n
        )


# -------------------------------
# Groq Explainable AI
# -------------------------------

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)


def explain(movie, rec):

    selected_row = final_movies[
        final_movies["title"].str.lower() == movie.lower()
    ]

    recommended_row = final_movies[
        final_movies["title"].str.lower() == rec.lower()
    ]

    if selected_row.empty or recommended_row.empty:
        return "Recommended because it matches your movie preferences."

    selected_tags = str(
        selected_row.iloc[0]["tags"]
    )

    recommended_tags = str(
        recommended_row.iloc[0]["tags"]
    )

    prompt = f"""
The user selected "{movie}".

Selected movie characteristics:
{selected_tags}

Recommended movie:
{rec}

Recommended movie characteristics:
{recommended_tags}

Give ONLY ONE short sentence, maximum 20 words.

Start with:
"Recommended because..."

The sentence should combine:
- similarity to the user's selected movie
- similar genres or themes
- audience preference when supported

Do not mention:
- movie names
- Content Score
- Collaborative Score
- Hybrid Score
- AI
- algorithms
- scores
- numbers

Do not invent information.

Example:
"Recommended because it shares similar genres and themes with your selected movie and matches your preferences."
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=50
    )

    return response.choices[0].message.content.strip()

# -------------------------------
# Sidebar
# -------------------------------

with st.sidebar:

    st.header("⚙ Recommendation Settings")

    movie = st.selectbox(
        "🎬 Select a Movie",
        options=sorted(final_movies["title"].unique())
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

tab1, tab2, tab3 = st.tabs(
    [
        "🎬 Recommendations",
        "📊 EDA Dashboard",
        "ℹ️ About Project"
    ]
)


# -------------------------------
# Recommendation Page
# -------------------------------

with tab1:

    if recommend:

        if not movie:
            st.warning("Please select a movie.")
            st.stop()

        with st.spinner("🔍 Finding the best movies for you..."):
            result = recommend_with_cold_start(
                user_id=user,
                movie_title=movie,
                top_n=top
            )

        if "Message" in result.columns:

            st.error(result["Message"].iloc[0])

        else:

            if is_new_user(user):
                st.info(
                    "🆕 New user detected. "
                    "Since you don't have previous rating history, "
                    "recommendations are based on the movie you selected."
                )
            else:
                st.success(
                    "👤 Existing user detected. "
                    "Recommendations are personalized using your "
                    "rating history and movie preferences."
                )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Movies Recommended",
                len(result)
            )

            col2.metric(
                "Selected Movie",
                movie
            )

            col3.metric(
                "Top Recommendation",
                result.iloc[0]["title"]
            )

            st.markdown("---")
            st.header("🎬 Recommended Movies")

            for i, (_, row) in enumerate(result.iterrows(), start=1):

                st.subheader(
                    f"{i}. 🎬 {row['title']}"
                )

                if is_new_user(user):
                    st.info(
                        "🎯 Recommended because it is similar "
                        "to the movie you selected."
                    )
                    st.caption(
                        "Cold-start recommendation based on your "
                        "initial movie preference."
                    )
                else:
                    st.info(
                        "👥 Recommended based on your previous "
                        "rating preferences and similar movie content."
                    )
                    st.caption(
                        "The hybrid model combines your learned "
                        "preferences with movie similarity."
                    )

                with st.expander("🤖 Why was this movie recommended?"):
                    try:
                        explanation = explain(
                            movie,
                            row["title"]
                        )
                        st.write(explanation)
                    except Exception:
                        st.write(
                            "Recommended because it shares similar "
                            "genres, themes, and audience preferences."
                        )

                st.divider()

# -------------------------------
# EDA Dashboard
# -------------------------------

def show_image(primary_path, fallback_path, caption):
    """Show an EDA image and support the two common filename spellings."""
    import os

    if os.path.exists(primary_path):
        image_path = primary_path
    elif os.path.exists(fallback_path):
        image_path = fallback_path
    else:
        st.warning(f"EDA image not found: {primary_path}")
        return

    st.image(
        image_path,
        caption=caption,
        width="stretch"
    )


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

    show_image(
        "images/movie_released_peryear.png",
        "images/movie_relased_peryear.png",
        "Movies Released Per Year"
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

# -------------------------------
# About Project
# -------------------------------

with tab3:

    st.title("🎬 About the Project")

    st.markdown("""
## AI Movie Recommendation System

This project recommends personalised movies using a Hybrid Recommendation System.

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

---

### 🧠 Models Used

#### 🎯 Content-Based Filtering

Uses **TF-IDF Vectorization** and **Cosine Similarity** to compare movie genres,
keywords, cast, and descriptions. It recommends movies with similar content.

---

#### 👥 Collaborative Filtering

Uses **Singular Value Decomposition (SVD)** from the Surprise library.

SVD is a **model-based collaborative filtering algorithm** that learns user
preferences from historical ratings and predicts ratings for unseen movies.

---

#### 🔀 Hybrid Recommendation

The final recommendation combines:

- 40% Content-Based Filtering
- 60% Collaborative Filtering (SVD)

This improves recommendation accuracy and personalization.

---

#### 🤖 Explainable AI

Groq Llama 3.3 generates natural language explanations describing why each
movie is recommended.
""")

    # -------------------------------
    # Model Evaluation
    # -------------------------------

    st.subheader("📈 Model Performance")

    metrics = pd.DataFrame({
        "Model": [
            "Collaborative (SVD)",
            "Hybrid"
        ],
        "Evaluation": [
            "RMSE / MAE",
            "Precision@10 & Recall@10"
        ],
        "Result": [
            "RMSE: 0.88 | MAE:0.68",
            "Precision:1.0 | Recall:0.0004"
        ]
    })

    st.table(metrics)

    # -------------------------------
    # Footer
    # -------------------------------

    st.markdown("---")

    st.markdown("""
### 👩‍💻 Developed by

**Prerana Gowda**

Business Analytics Student

AI & Machine Learning Enthusiast

---

### 🎬 AI Movie Recommendation System

✔ Content-Based Filtering

✔ Collaborative Filtering (SVD)

✔ Hybrid Recommendation

✔ Explainable AI
""")
