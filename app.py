import streamlit as st
import pandas as pd
import joblib

# ==========================================
# Load Model, Features and Dataset
# ==========================================

model = joblib.load("movie_recommendation_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
df = pd.read_csv("df.csv")

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")

st.write(
    "Select your preferences to receive movie recommendations using Machine Learning."
)

# ==========================================
# User Inputs
# ==========================================

genres = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "IMAX",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "Unknown",
    "War",
    "Western"
]

selected_genre = st.selectbox(
    "🎭 Select Genre",
    genres
)

years = sorted(
    df["release_year"]
    .dropna()
    .astype(int)
    .unique(),
    reverse=True
)

selected_year = st.selectbox(
    "📅 Minimum Release Year",
    years,
    index=0
)

top_n = st.selectbox(
    "🎯 Number of Recommendations",
    [5, 10, 15, 20],
    index=1
)

# ==========================================
# Recommendation
# ==========================================

if st.button("Recommend Movies"):

    movies = df.copy()

    # Filter by selected genre
    movies = movies[movies[selected_genre] == 1]

    # Filter by release year
    movies = movies[
        movies["release_year"] >= selected_year
    ]

    if movies.empty:

        st.warning("No movies found for the selected criteria.")

    else:

        # Build feature matrix using exactly the
        # same columns used during training
        X = movies.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # Predict ratings
        movies["Predicted Rating"] = model.predict(X)

        # Sort movies
        movies = movies.sort_values(
            by="Predicted Rating",
            ascending=False
        )

        # Display results
        st.success(f"Top {top_n} Recommended Movies")

        result = movies[
            [
                "title",
                "genres",
                "release_year",
                "Predicted Rating"
            ]
        ].head(top_n)

        result["Predicted Rating"] = result[
            "Predicted Rating"
        ].round(2)

        st.dataframe(
            result.reset_index(drop=True),
            use_container_width=True
        )