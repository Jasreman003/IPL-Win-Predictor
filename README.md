# 🏏 IPL Win Probability Predictor

A Machine Learning-powered web application that predicts the winning probability of an IPL team during a live match based on current match conditions such as score, overs, wickets, target score, teams, and venue.

Built using **Python, Scikit-learn, Streamlit, Pandas, NumPy, and IPL Ball-by-Ball Data**.

---

## 🚀 Live Demo

👉 https://jasreman003-ipl-win-predictor-app-idvfqa.streamlit.app/

---

## 📌 Project Overview

This project predicts the probability of the batting team winning an IPL match in real time.

The model analyzes:

* Batting Team
* Bowling Team
* Venue
* Current Score
* Target Score
* Overs Completed
* Balls Remaining
* Required Run Rate
* Current Run Rate

and generates winning probabilities for both teams.

---

## 🎯 Features

✅ Real-time IPL win probability prediction

✅ Interactive Streamlit dashboard

✅ Machine Learning-based prediction engine

✅ Logistic Regression model

✅ Venue-based prediction

✅ Team-wise prediction

✅ Match scenario explorer

✅ Beautiful modern UI

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Plotly
* Pickle

### Machine Learning

* Logistic Regression

### Dataset

IPL Ball-by-Ball Dataset (Kaggle)

---

## 📂 Project Structure

```bash
IPL-Win-Predictor/
│
├── .devcontainer/
│   └── devcontainer.json
│
├── data/
│   ├── deliveries.csv
│   ├── matches.csv
│   └── README.md
│
├── images/
│   ├── App Overview.png
│   ├── App Sidebar.png
│   ├── Confusion Matrix.png
│   ├── Correlation Heatmap.png
│   ├── Feature Importance.png
│   ├── Live Match Stats.png
│   ├── Match State Input.png
│   ├── Match Winners Distribution.png
│   ├── Model Comparison.png
│   ├── Most Successful IPL Teams.png
│   ├── Scenario Explorer.png
│   └── Target Score Distribution.png
│
├── model/
│   ├── ipl_model.pkl
│   ├── encoders.pkl
│   └── README.md
│
├── notebooks/
│   └── analysis.ipynb
│
├── .gitignore
├── README.md
├── app.py
├── requirements.txt
└── train_model.py
```

---

## 📊 Dataset

Dataset used:

IPL Complete Dataset (2008–2020)

Download from Kaggle:

https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020

After downloading:

1. Create a data folder
2. Place:

   * deliveries.csv
   * matches.csv

inside the folder.

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/IPL-Win-Predictor.git

cd IPL-Win-Predictor
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python train_model.py
```

This creates:

```bash
model/ipl_model.pkl

model/encoders.pkl
```

### Run Application

```bash
streamlit run app.py
```

---

## 📈 Model Performance

| Metric            | Score |
| ----------------- | ----- |
| Accuracy          | 79.11% |
| ROC-AUC           | 0.846 |
| Baseline Accuracy | 61.89% |
| Improvement       | +7.3% |

---

## 📱 Application Screenshots

### Home Dashboard

#### App Overview
![App Overview](images/App%20Overview.png)

#### Sidebar
![App Sidebar](images/App%20Sidebar.png)

#### Match State Input
![Match State Input](images/Match%20State%20Input.png)

#### Live Match Stats
![Live Match Stats](images/Live%20Match%20Stats.png)

#### Scenario Explorer
![Scenario Explorer](images/Scenario%20Explorer.png)

---

## 📊 Exploratory Data Analysis

### Most Successful IPL Teams
![Most Successful IPL Teams](images/Most%20Successful%20IPL%20Teams.png)

### Match Winners Distribution
![Match Winners Distribution](images/Match%20Winners%20Distribution.png)

### Target Score Distribution
![Target Score Distribution](images/Target%20Score%20Distribution.png)

### Correlation Heatmap
![Correlation Heatmap](images/Correlation%20Heatmap.png)

### Model Comparison
![Model Comparison](images/Model%20Comparison.png)

### Confusion Matrix
![Confusion Matrix](images/Confusion%20Matrix.png)

### Feature Importance
![Feature Importance](images/Feature%20Importance.png)

---

## 🔮 Future Improvements

* Random Forest Model
* XGBoost Model
* Team Head-to-Head Analysis
* Win Probability Graph
* Player Statistics
* Live Match API Integration
* Match Simulation Engine

---

## 👩‍💻 Author

### Jasreman Kaur

BCA (Artificial Intelligence)

Data Analyst | Machine Learning Enthusiast

GitHub: https://github.com/Jasreman003

LinkedIn: https://www.linkedin.com/in/jasreman-kaur-818568298 

---

## ⭐ Support

If you found this project useful:

⭐ Star this repository

🍴 Fork this repository

📢 Share with others
