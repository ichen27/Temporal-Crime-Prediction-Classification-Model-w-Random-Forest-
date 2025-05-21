#🕵️ Temporal Crime Prediction Classification Model using Random Forest
Predict the type of crime most likely to occur based on temporal features such as year, month, day, and weekday using a Random Forest Classifier. Built using real-world crime incident data from Syracuse, NY, this project explores temporal trends in crime and builds a machine learning model for predictive insights.


##🧠 Overview

This project classifies crimes based solely on when they occurred, aiming to understand if specific crime types have predictable temporal patterns. It preprocesses timestamped crime reports, extracts temporal features, and trains a Random Forest model to predict the most likely crime type for any given date.

##✅ Features

Cleaned and structured 40,000+ crime reports from 2020–2024
Extracted month, day of week, day of month, and year from datetime fields
Trained and evaluated multiple classification models
Built a prediction function to input a date and return crime probabilities
Stored processed statistics in SQLite for long-term querying and analysis


##⚙️ Technologies Used

Python (pandas, scikit-learn, numpy, sqlite3)
Scikit-learn for model training and evaluation
SQLite for storing summarized statistics
Matplotlib for EDA visualizations (optional)


##📂 Dataset

Source: Publicly available crime reports from Syracuse, NY
Fields used:
DATEEND – Date of the incident
CODE_DEFINED – Crime category (e.g., LARCENY, ROBBERY, MURDER)
