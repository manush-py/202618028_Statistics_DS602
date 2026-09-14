# Lab-4: Applied Statistical Modeling & Interactive Web Dashboard

**Live Application URL:** https://202618028-ds602-statistics-mdoshi.streamlit.app/

### Student & Course Information
* **Name:** Manush Doshi
* **Roll Number:** 202618028
* **Program:** M.Sc. Data Science (Semester 1)
* **Course:** DS602 — Applied Statistics

---

## Project Overview
This repository contains an end-to-end data science project spanning exploratory data analysis (EDA), rigorous statistical hypothesis testing, and Ordinary Least Squares (OLS) regression modeling. The mathematical models and inferential tests are deployed as a highly interactive web application using Streamlit, allowing users to dynamically filter data, run tests, and generate real-time predictions with diagnostic checks.

## Dataset Summary
The application utilizes the `insurance.csv` dataset, which contains medical billing data for ,1338 patients. 
* **Target Variable:** `charges` (Continuous - Individual medical costs billed by health insurance).
* **Numerical Predictors:** `age` (Years), `bmi` (Body Mass Index), `children` (Number of dependents).
* **Categorical Predictors:** `sex` (Male/Female), `smoker` (Yes/No), `region` (Northeast, Northwest, Southeast, Southwest).

## Synthesis of Statistical Findings

1. **Exploratory Data Analysis (EDA):**
   * Medical charges exhibit a severe rightward skew (Skewness: ~1.52) with high kurtosis, indicating a small segment of patients incurs exceptionally high medical costs.
   * Bivariate scatter plots reveal distinct clustering: smokers consistently face significantly higher baseline charges than non-smokers, regardless of age.

2. **Hypothesis Testing:**
   * **Test 1 (Continuous Metric Comparison):** A Mann-Whitney U test was utilized (due to violated normality and variance assumptions) to compare charges between smokers and non-smokers. The test yielded a p-value strictly `< 0.05`, leading to the rejection of the null hypothesis. There is a statistically significant difference in charges based on smoking status.
   * **Test 2 (Categorical Independence):** A Chi-Square Test of Independence was conducted between smoking status and geographic region. [Update based on your app's output: typically fails to reject $H_0$, meaning smoking status is independent of region].

3. **OLS Regression Modeling & Diagnostics:**
   * A multiple linear regression model was fit to predict `charges`. Smoking status (`smoker_yes`), `age`, and `bmi` emerged as the most highly significant predictors (p < 0.05).
   * **Gauss-Markov Diagnostics:** Variance Inflation Factors (VIF) for continuous variables remained well below the threshold of 5, indicating no severe multicollinearity. However, Q-Q plots of the residuals indicate a deviation from perfect normality at the upper tail, mirroring the right-skewed nature of the target variable.

---

## How to Run the Application Locally

### Prerequisites
Ensure you have Python 3.9+ installed.

### 1. Clone the Repository
```bash
git clone https://github.com/manush-py/202618028_Statistics_DS602.git
cd 202618028_Statistics_DS602
```

### 2. Install Dependencies
It is recommended to use a virtual environment. Install the required packages using the provided requirements file:
```bash
pip install -r 202618028_Lab_4/requirements.txt
```

### 3. Launch the Streamlit Dashboard
Because the dataset and app are nested in a subfolder, run the following command from the root of the repository:
```bash
streamlit run 202618028_Lab_4/app.py
```
*The application will open automatically in your default web browser at `http://localhost:8501`.*