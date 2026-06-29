# 🎬 Movie Recommendation System

## 📌 Overview

This project is the **final project** for the **Data Science Course** at **Epsilon Academy**.

The project presents a **Movie Recommendation System** that leverages **Machine Learning** to predict movie ratings based on movie characteristics and recommend the highest-rated movies. It demonstrates the complete Data Science workflow, from data exploration and preprocessing to model deployment.

---

## 🎯 Project Objectives

- Analyze movie data to extract meaningful insights.
- Build a machine learning model capable of predicting movie ratings.
- Recommend the best movies based on the predicted ratings.
- Deploy the trained model as an interactive web application using **Streamlit**.

---

## 📂 Dataset

The project uses the **MovieLens Dataset**, which includes:

- **Movies Dataset:** Movie titles and genres.
- **Links Dataset:** IMDb and TMDb identifiers.
- **Tags Dataset:** User-generated movie tags.
- **Ratings Dataset:** User ratings used for training the machine learning model.

The **movies**, **links**, and **tags** datasets were merged into a single dataframe, while the **ratings** dataset was used as the target for model training.

---

## 📊 Model Evaluation

The model was evaluated using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score (Coefficient of Determination)

After comparing several machine learning algorithms, **XGBoost** achieved the best performance and was selected as the final model.

---

## 🚀 Deployment

The final recommendation system was deployed as an interactive web application using **Streamlit**.

Users can:

- Select their preferred movie genre.
- Choose a minimum release year.
- Specify the number of recommendations.
- Receive the top recommended movies ranked by the predicted rating.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- Joblib

---

## 📸 Application

The Streamlit application provides an interactive interface where users can receive personalized movie recommendations based on their selected preferences.

---

## 🙏 Acknowledgments

I would like to express my sincere gratitude to **Epsilon Academy** for providing an excellent learning experience throughout the Data Science course.

Special thanks to **Eng. Salah** for his continuous guidance, support, and valuable feedback throughout the project. 

https://github.com/EPSILON-AI
