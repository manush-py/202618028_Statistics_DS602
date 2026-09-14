import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.express as px

st.set_page_config(page_title="Lab-4: Statistical Modeling", layout="wide")
st.title("Applied Statistical Modeling & Interactive Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset not found. Please ensure 'insurance.csv' is in the directory.")
    st.stop()

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# --- TABS SETUP ---
tab1, tab2, tab3 = st.tabs(["Data Exploration", "Hypothesis Testing Lab", "Live Prediction & Diagnostics"])

# ==========================================
# TAB 1: DATA EXPLORATION
# ==========================================
with tab1:
    st.sidebar.header("Interactive Data Filters")
    
    # Sidebar Filters
    age_range = st.sidebar.slider("Select Age Range", int(df['age'].min()), int(df['age'].max()), (18, 64))
    smoker_filter = st.sidebar.multiselect("Select Smoker Status", options=df['smoker'].unique(), default=df['smoker'].unique())
    
    filtered_df = df[(df['age'] >= age_range[0]) & 
                     (df['age'] <= age_range[1]) & 
                     (df['smoker'].isin(smoker_filter))]
    
    st.subheader("1. Descriptive Metrics")
    if not filtered_df.empty:
        stats_df = filtered_df[num_cols].describe().T
        stats_df['IQR'] = stats_df['75%'] - stats_df['25%']
        stats_df['Skewness'] = filtered_df[num_cols].skew()
        stats_df['Kurtosis'] = filtered_df[num_cols].kurtosis()
        
        display_stats = stats_df[['mean', '50%', 'std', 'IQR', 'Skewness', 'Kurtosis']].rename(columns={'50%': 'Median'})
        st.dataframe(display_stats.style.format("{:.2f}"))
    else:
        st.warning("Filters are too strict. No data available.")

    st.subheader("2. Visual Exploration")
    eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Distribution Plot", "Bivariate Scatter", "Correlation Matrix"])
    
    # 1. Distribution of Charges (Interactive)
    with eda_tab1:
        st.markdown("**Distribution of Charges (Histogram & Marginal Boxplot)**")
        fig_hist = px.histogram(filtered_df, x='charges', color="smoker", marginal="box", 
                                hover_data=df.columns, title="Distribution of Medical Charges")
        st.plotly_chart(fig_hist, use_container_width=True)

    # 2. Age vs Charges by Smoker (Interactive)
    with eda_tab2:
        st.markdown("**Age vs Charges (Separated by Smoker Status)**")
        fig_scatter = px.scatter(filtered_df, x='age', y='charges', color="smoker", 
                                 size="bmi", hover_data=['region', 'children'],
                                 title="Age vs Charges by Smoker Status")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # 3. Correlation Matrix (Interactive)
    with eda_tab3:
        st.markdown("**Correlation Matrix**")
        corr_matrix = filtered_df[num_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", 
                             color_continuous_scale="RdBu_r", title="Feature Correlation Heatmap")
        st.plotly_chart(fig_corr, use_container_width=True)

# ==========================================
# TAB 2: HYPOTHESIS TESTING LAB
# ==========================================
with tab2:
    st.header("Hypothesis Testing Laboratory")
    
    st.subheader("Test 1: Compare Two Groups (Independent Samples)")
    col_group, col_metric = st.columns(2)
    with col_group:
        group_var = st.selectbox("Select Binary Categorical Variable", ['smoker', 'sex'])
    with col_metric:
        metric_var = st.selectbox("Select Continuous Metric", num_cols, index=num_cols.index('charges'))
        
    groups = df[group_var].unique()
    if len(groups) == 2:
        group1 = df[df[group_var] == groups[0]][metric_var]
        group2 = df[df[group_var] == groups[1]][metric_var]
        
        st.markdown(f"**H0**: No difference in {metric_var} between {groups[0]} and {groups[1]}.")
        st.markdown(f"**H1**: A significant difference exists.")
        
        # Normality & Variance Checks
        _, p_sw1 = stats.shapiro(group1)
        _, p_sw2 = stats.shapiro(group2)
        _, p_lev = stats.levene(group1, group2)
        
        if p_sw1 > 0.05 and p_sw2 > 0.05 and p_lev > 0.05:
            stat, p_val = stats.ttest_ind(group1, group2)
            test_used = "Independent T-Test"
        else:
            stat, p_val = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            test_used = "Mann-Whitney U Test"
            
        st.write(f"**Test Utilized**: {test_used}")
        st.write(f"**p-value**: {p_val:.4e}")
        
        if p_val < 0.05:
            st.error(f"**Conclusion**: Reject H0 at α = 0.05. There is a significant difference.")
        else:
            st.success(f"**Conclusion**: Fail to Reject H0 at α = 0.05.")
            
    st.divider()
    
    st.subheader("Test 2: Chi-Square Test of Independence")
    col_cat1, col_cat2 = st.columns(2)
    with col_cat1:
        cat1 = st.selectbox("Select Categorical Variable 1", cat_cols, index=0)
    with col_cat2:
        cat2 = st.selectbox("Select Categorical Variable 2", cat_cols, index=2)
        
    if cat1 != cat2:
        st.markdown(f"**H0**: {cat1} and {cat2} are independent.")
        st.markdown(f"**H1**: {cat1} and {cat2} are dependent/linked.")
        
        contingency = pd.crosstab(df[cat1], df[cat2])
        chi2, p_chi2, dof, exp = stats.chi2_contingency(contingency)
        
        st.write(f"**p-value**: {p_chi2:.4f}")
        if p_chi2 < 0.05:
            st.error("**Conclusion**: Reject H0 at α = 0.05. Variables are significantly linked.")
        else:
            st.success("**Conclusion**: Fail to Reject H0 at α = 0.05. Variables are independent.")
    else:
        st.warning("Select two distinct categorical variables.")

# ==========================================
# TAB 3: LIVE PREDICTION & DIAGNOSTICS
# ==========================================
with tab3:
    st.header("Interactive OLS Model & Diagnostics")
    
    # 1. Fit Background OLS Model
    df_encoded = pd.get_dummies(df, drop_first=True, dtype=int)
    X = sm.add_constant(df_encoded.drop('charges', axis=1))
    Y = df_encoded['charges']
    model = sm.OLS(Y, X).fit()
    
    # 2. Live Prediction Engine
    st.subheader("Real-Time Prediction Tool")
    c1, c2, c3, c4 = st.columns(4)
    pred_age = c1.number_input("Age", min_value=18, max_value=100, value=30)
    pred_bmi = c2.number_input("BMI", min_value=15.0, max_value=55.0, value=25.0)
    pred_child = c3.number_input("Children", min_value=0, max_value=10, value=0)
    pred_sex = c4.selectbox("Sex", ["male", "female"])
    
    c5, c6 = st.columns(2)
    pred_smoker = c5.selectbox("Smoker", ["yes", "no"])
    pred_region = c6.selectbox("Region", df['region'].unique())
        
    input_df = pd.DataFrame({
        'const': 1,
        'age': [pred_age],
        'bmi': [pred_bmi],
        'children': [pred_child],
        'sex_male': [1 if pred_sex == 'male' else 0],
        'smoker_yes': [1 if pred_smoker == 'yes' else 0],
        'region_northwest': [1 if pred_region == 'northwest' else 0],
        'region_southeast': [1 if pred_region == 'southeast' else 0],
        'region_southwest': [1 if pred_region == 'southwest' else 0]
    })
    
    input_df = input_df[X.columns]
    prediction = model.get_prediction(input_df)
    pred_summary = prediction.summary_frame(alpha=0.05)
    
    st.success(f"### Predicted Charge: ${pred_summary['mean'].values[0]:,.2f}")
    st.markdown(f"**95% Confidence Interval**: ${pred_summary['mean_ci_lower'].values[0]:,.2f} to ${pred_summary['mean_ci_upper'].values[0]:,.2f}")
    
    st.divider()
    
    # 3. Interactive Diagnostic Plots (Plotly)
    st.subheader("Residual Diagnostic Plots")
    residuals = model.resid
    fitted = model.fittedvalues
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("**1. Residuals vs. Fitted (Homoscedasticity)**")
        fig_resid = px.scatter(x=fitted, y=residuals, trendline="lowess", 
                               trendline_color_override="red", opacity=0.6,
                               labels={'x': 'Fitted Values', 'y': 'Residuals'})
        fig_resid.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_resid, use_container_width=True)
        
    with col_d2:
        st.markdown("**2. Q-Q Plot (Normality)**")
        osm, osr = stats.probplot(residuals, dist="norm", fit=False)
        fig_qq = px.scatter(x=osm, y=osr, opacity=0.6, labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'})
        fig_qq.add_shape(type="line", line=dict(color="red", dash="dash"),
                         x0=min(osm), y0=min(osr), x1=max(osm), y1=max(osr))
        st.plotly_chart(fig_qq, use_container_width=True)
        
    # 4. Multicollinearity VIF Table
    st.markdown("**3. Variance Inflation Factors (VIF)**")
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    st.dataframe(vif_data[vif_data['Feature'] != 'const'].style.format({"VIF": "{:.2f}"}), use_container_width=True)