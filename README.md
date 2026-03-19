# Air-Quality-ML-prediction

# 🌍 Air Quality Prediction using Machine Learning

📌 Overview

This project develops a machine learning pipeline to predict Relative Humidity (RH) using air quality sensor measurements.

The study investigates how environmental pollutants and temporal factors influence humidity and evaluates multiple regression models to identify the most effective approach.

🚀 Key Highlights

End-to-end machine learning pipeline in MATLAB

Time-aware missing value imputation

Feature engineering using temporal data

Comparison of multiple regression models

---

## 📊 Dataset

The dataset contains air quality measurements including:

* Date:	The date of the measurement.
* Time:	The time of the measurement.
* CO(GT):	Concentration of carbon monoxide (CO) in the air (µg/m³).
* PT08.S1(CO):	Sensor measurement for CO concentration.
* NMHC(GT):	Concentration of non-methane hydrocarbons (NMHC) (µg/m³).
* C6H6(GT):	Concentration of benzene (C6H6) in the air (µg/m³).
* PT08.S2(NMHC):	Sensor measurement for NMHC concentration.
* NOx(GT):	Concentration of nitrogen oxides (NOx) in the air (µg/m³).
* PT08.S3(NOx):	Sensor measurement for NOx concentration.
* NO2(GT):	Concentration of nitrogen dioxide (NO2) in the air (µg/m³).
* Temperature (T)
* Relative Humidity (RH) → **Target variable**

### ⚠️ Important

Missing values are represented as **-200**

---

## ⚙️ Methodology

### 1. Data Preprocessing

* Converted data to numeric format
* Replaced missing values (-200) using **hourly mean**
* Removed highly missing feature (NMHC)

---

### 2. Feature Engineering

Extracted time-based features:

* Hour
* Month
* Day of week

---

### 3. Models Used

* Linear Regression (baseline)
* Decision Tree Regression
* Optimized Decision Tree (Grid Search)
* Random Forest (Bayesian Optimization)

---

## 📈 Results

* Linear Regression performs poorly due to weak linear relationships
* Decision Trees improve performance by capturing non-linearity
* Random Forest gives the **best performance and stability**

(*Numeric results will be added later*)

---

## 🧠 Key Learnings

* Environmental data is highly non-linear
* Proper handling of missing values is critical
* Ensemble models perform best for this task

---

## 🚀 How to Run

1. Download dataset (see `data/README.md`)
2. Update file path in `main.m`
3. Run:

```matlab
main
```

---

## 🔮 Future Work

* Apply LSTM / GRU models
* Explore Transformer-based models
* Improve temporal modeling
