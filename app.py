import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model and Dataset
# -----------------------------

model = joblib.load("movie_recommendation_model.pkl")

df = pd.read_csv("df.csv")

# -----------------------------
# Streamlit Page
# -----------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")

st.write(
    "Select your preferences to receive movie recommendations based on our machine learning model."
)

# -----------------------------
# User Inputs
# -----------------------------

all_genres = sorted(
    df["genres"].dropna().unique()
)

selected_genre = st.selectbox(
    "Select Genre",
    all_genres
)

min_year = st.slider(
    "Minimum Release Year",
    int(df["release_year"].min()),
    int(df["release_year"].max()),
    2015
)

top_n = st.slider(
    "Number of Recommendations",
    5,
    20,
    10
)

# -----------------------------
# Recommend Movies
# -----------------------------

if st.button("Recommend Movies"):

    movies = df.copy()

    movies = movies[
        movies["genres"].str.contains(
            selected_genre,
            case=False,
            na=False
        )
    ]

    movies = movies[
        movies["release_year"] >= min_year
    ]

    if movies.empty:

        st.warning("No movies found.")

    else:

        # -----------------------------
        # Features used during training
        # -----------------------------

        genre_features = [

            "(no genres listed)",
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

        features = [

            "release_year",
            "movie_age",
            "genre_count",
            "tag_count",
            "title_length",
            "has_tag",
            "is_multi_genre",
            "decade"

        ] + genre_features

        X = movies[features].fillna(0)

        # -----------------------------
        # Prediction
        # -----------------------------

        movies["Predicted Rating"] = model.predict(X)

        movies = movies.sort_values(
            by="Predicted Rating",
            ascending=False
        )

        # -----------------------------
        # Display Results
        # -----------------------------

        st.success(f"Top {top_n} Recommended Movies")

        st.dataframe(

            movies[
                [
                    "title",
                    "genres",
                    "release_year",
                    "Predicted Rating"
                ]
            ]
            .head(top_n)
            .reset_index(drop=True),

            use_container_width=True

        )