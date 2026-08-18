# Customer Satisfaction Analysis & Machine Learning Pipeline

An end-to-end data science project analyzing customer support ticket metrics to identify key drivers of customer satisfaction (CSAT) scores, validate operational hypotheses using inferential statistics, and predict customer ratings using machine learning models.

---

## 📌 Executive Summary

* **Primary Driver:** Support response time (`Response_Time_Mins`) is the single most critical factor influencing CSAT scores (~50% feature importance).
* **Secondary Drivers:** Product item price (`Item_price`) and support channel choice significantly impact overall satisfaction ratings.
* **Statistical Insights:** Significant CSAT variance was confirmed across support channels ($p \le 0.05$) using One-Way ANOVA tests.

---

## 🛠 Project Architecture & Workflow

1. **Environment & Setup:** Isolated Python environment (`venv`) with `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn`.
2. **Data Ingestion & Cleaning:**
   * Handled missing ratings and imputed numerical missing values with medians.
   * Engineered `Response_Time_Mins` feature from datetime resolution timestamps.
3. **Exploratory Data Analysis (EDA):**
   * Target variable (`CSAT Score`) distribution mapping.
   * Grouped performance analysis across support channels and agent shifts.
   * Pearson correlation matrix visualization.
4. **Hypothesis Testing:**
   * Welch's Two-Sample T-Test across shift categories.
   * One-Way ANOVA across support channels.
5. **Machine Learning Pipeline:**
   * Categorical feature encoding using One-Hot Encoding (`pd.get_dummies`).
   * 80/20 train-test split setup.
   * Model training: **Linear Regression** and **Random Forest Regressor**.
   * Model evaluation using MAE, RMSE, and $R^2$ Score.
   * Feature importance extraction via Random Forest.

---

## 📊 Key Results & Model Performance

| Model | MAE | RMSE | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | *Evaluated* | *Evaluated* | Baseline |
| **Random Forest Regressor** | **Optimal** | **Optimal** | **Best Fit** |

---

