import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("movie_recommendation_model.pkl")

# Load dataset
df = pd.read_csv("df.csv")

st.title("🎬 Movie Recommendation System")

st.write("Select your preferences and get movie recommendations.")

# -------------------
# User Inputs
# -------------------

genre = st.selectbox(

    "Genre",

    sorted(df["genres"].dropna().unique())

)

year = st.slider(

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

# -------------------
# Recommendation
# -------------------

if st.button("Recommend Movies"):

    movies = df.copy()

    movies = movies[

        movies["genres"].str.contains(

            genre,

            case=False,

            na=False

        )

    ]

    movies = movies[

        movies["release_year"] >= year

    ]

    if len(movies) == 0:

        st.warning("No movies found.")

    else:

        features = [

            "release_year",

            "movie_age",

            "genre_count",

            "tag_count",

            "title_length",

            "has_tag",

            "is_multi_genre",

            "decade"

        ]

        X = movies[features].fillna(0)

        movies["Predicted Rating"] = model.predict(X)

        movies = movies.sort_values(

            "Predicted Rating",

            ascending=False

        )

        st.success("Top Recommended Movies")

        st.dataframe(

            movies[

                [

                    "title",

                    "genres",

                    "Predicted Rating"

                ]

            ].head(top_n)

        )