import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os

# Set page config
st.set_page_config(
    page_title="Ethical Aggregate HR Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #fff;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f0fa;
        color: #1f77b4;
        font-weight: 600;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_data():
    # Load models
    calib_path = '../models/calibrated_lightgbm_model.pkl'
    uncalib_path = '../models/best_lightgbm_model.pkl'
    if not os.path.exists(calib_path):
        calib_path = 'models/calibrated_lightgbm_model.pkl'
        uncalib_path = 'models/best_lightgbm_model.pkl' # Run from root
    calib_model = joblib.load(calib_path)
    uncalib_model = joblib.load(uncalib_path)
    
    # Load data
    data_path = '../data/fevs_processed_for_ml.csv'
    if not os.path.exists(data_path):
        data_path = 'data/fevs_processed_for_ml.csv'
    df = pd.read_csv(data_path)
    
    # Load codebook
    codebook_path = '../data/codebook_extracted.csv'
    if not os.path.exists(codebook_path):
        codebook_path = 'data/codebook_extracted.csv'
    codebook = pd.read_csv(codebook_path)
    
    return calib_model, uncalib_model, df, codebook

def prepare_data(df, codebook):
    y = df['Turnover_Intention']
    X = df.drop(columns=['Turnover_Intention', 'POSTWT'])
    X_num = X.apply(pd.to_numeric, errors='coerce')
    X_filled = X_num.fillna(-1)
    
    # Calculate initial probabilities for everyone
    probs_full = model.predict_proba(X_filled)[:, 1]
    
    # Sample data to avoid excessive computation time for SHAP
    X_sample = X_filled.sample(n=1000, random_state=42)
    
    # Create mapping
    cb_dict = dict(zip(codebook['VARIABLE'], codebook['VARIABLE'] + ": " + codebook['ITEM TEXT']))
    
    # Add details for Q15_1 - Q15_6
    q15_suffixes = {
        'Q15_1': 'Remain in the work unit and improve their performance over time',
        'Q15_2': 'Remain in the work unit and continue to underperform',
        'Q15_3': 'Leave the work unit - removed or transferred',
        'Q15_4': 'Leave the work unit - quit',
        'Q15_5': 'There are no poor performers in my work unit',
        'Q15_6': 'Do Not Know'
    }
    for q, suffix in q15_suffixes.items():
        cb_dict[q] = f"{q}: In my work unit poor performers usually: {suffix}"
        
    # Ensure specific variables have complete descriptions
    if 'Q43' not in cb_dict: cb_dict['Q43'] = "Q43: I recommend my organization as a good place to work."
    if 'Q61' not in cb_dict: cb_dict['Q61'] = "Q61: Senior leaders demonstrate support for Work-Life programs."
    if 'Q68' not in cb_dict: cb_dict['Q68'] = "Q68: Considering everything, how satisfied are you with your job?"
    if 'Q69' not in cb_dict: cb_dict['Q69'] = "Q69: Considering everything, how satisfied are you with your pay?"
    if 'Q90' not in cb_dict: cb_dict['Q90'] = "Q90: What percentage of your work time are you currently required to be physically present at your agency worksite (including headquarters, bureau, field offices, etc.)?"
    if 'Q91' not in cb_dict: cb_dict['Q91'] = "Q91: Please select the response that BEST describes your current remote work or teleworking schedule."
    if 'Q94' not in cb_dict: cb_dict['Q94'] = "Q94: My agency’s re-entry arrangements are fair in accounting for employees’ diverse needs and situations."

    feature_labels = [cb_dict.get(col, col) for col in X_sample.columns]
    
    return X_filled, X_sample, y, feature_labels, probs_full, cb_dict

# App Header
st.title("🏢 Ethical Aggregate HR Dashboard")
st.markdown("**Explainable Prediction of Public-Sector Employee Turnover Intention**")
st.markdown("This dashboard is designed on the principles of Data Ethics, presenting turnover risk at an aggregate level rather than identifying individuals, to serve as a guide for formulating personnel development policies.")
st.divider()

# Load Data
with st.spinner("Loading model and processing data..."):
    calib_model, uncalib_model, df, codebook = load_model_and_data()
    model = calib_model # Keep model pointing to the calibrated one for prepare_data
    X_full, X_sample, y_overall, feature_labels, initial_probs, cb_dict = prepare_data(df, codebook)

# Create Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Executive Overview", "🧠 Explainable AI Insights", "🔮 Policy Simulation (What-If)", "📚 Questionnaire Directory"])

# Tab 1: Executive Overview
with tab1:
    st.header("Aggregate Risk Overview")
    
    col1, col2, col3 = st.columns(3)
    
    avg_risk = np.mean(initial_probs) * 100
    high_risk_pct = (np.sum(initial_probs > 0.5) / len(initial_probs)) * 100
    
    with col1:
        st.markdown(f'<div class="metric-card"><h3>Total Employees in Database</h3><h1 style="color:#1f77b4;">{len(df):,}</h1><p>Employees</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>Average Aggregate Turnover Risk</h3><h1 style="color:#ff7f0e;">{avg_risk:.2f}%</h1><p>Average Probability</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h3>Proportion of High-Risk Employees (Risk > 50%)</h3><h1 style="color:#d62728;">{high_risk_pct:.2f}%</h1><p>of Organization</p></div>', unsafe_allow_html=True)
        
    st.markdown("### Turnover Risk Probability Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(initial_probs, bins=50, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(initial_probs), color='red', linestyle='dashed', linewidth=2, label=f'Mean Risk: {avg_risk:.2f}%')
    ax.set_xlabel("Probability of Turnover Intention")
    ax.set_ylabel("Number of Employees (per risk bin)")
    ax.legend()
    st.pyplot(fig)

# Tab 2: Explainable AI Insights
with tab2:
    st.header("Key Factors Influencing Turnover Intention (SHAP Global Explanation)")
    
    with st.spinner("Calculating SHAP Values... (This may take a moment)"):
        explainer = shap.TreeExplainer(uncalib_model)
        shap_values = explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list):
            shap_vals_to_plot = shap_values[1] # Class 1
        else:
            shap_vals_to_plot = shap_values
            
        # Filter out demographic features to only show questions
        exclude_cols = ['DDIS', 'DSEX', 'DAGEGRP', 'DSUPER', 'DFEDTEN', 'DRNO', 'DHISP', 'DMIL', 'AGENCY', 'AGCYTYP', 'RANDOM_ID']
        keep_indices = [i for i, col in enumerate(X_sample.columns) if col not in exclude_cols]
        
        shap_vals_filtered = shap_vals_to_plot[:, keep_indices]
        X_sample_filtered = X_sample.iloc[:, keep_indices]
        feature_labels_filtered = [feature_labels[i] for i in keep_indices]
            
        st.subheader("🔴 Top 15 Features")
        st.markdown("The graph displays the **top 15 parameters** most prioritized by the AI network (filtered to include only questions, excluding demographic data).")
        fig_top, ax_top = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_vals_filtered, X_sample_filtered, feature_names=feature_labels_filtered, max_display=15, show=False)
        st.pyplot(fig_top)
        st.info("💡 **How to read the graph:** Each point represents one employee. If a point falls to the right of the 0.0 line, it contributes to **increasing the risk** of the employee turning over.")

        st.divider()
        st.subheader("🟢 Bottom 15 Features")
        st.markdown("The graph displays the **bottom 15 parameters** least prioritized by the AI network (filtered to include only questions).")
        
        # Calculate mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(shap_vals_filtered).mean(axis=0)
        # Get indices of bottom 15 features
        bottom_15_indices = np.argsort(mean_abs_shap)[:15]
        
        # Extract corresponding SHAP values, feature values, and labels
        shap_vals_bottom = shap_vals_filtered[:, bottom_15_indices]
        X_sample_bottom = X_sample_filtered.iloc[:, bottom_15_indices]
        feature_labels_bottom = [feature_labels_filtered[i] for i in bottom_15_indices]
        
        fig_bottom, ax_bottom = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_vals_bottom, X_sample_bottom, feature_names=feature_labels_bottom, max_display=15, show=False)
        st.pyplot(fig_bottom)

    st.divider()
    st.subheader("🔍 Deep Dive into Feature & Risk Analysis")
    st.markdown("Select a parameter of interest to view a graph showing the average turnover risk (Probability) compared to the score level of that parameter across all personnel.")
    
    question_cols_tab2 = [col for col in X_full.columns if col.startswith('Q')]
    question_options_tab2 = {col: cb_dict.get(col, col) for col in question_cols_tab2}
    
    selected_feature = st.selectbox(
        "Select parameter to analyze:",
        options=question_cols_tab2,
        format_func=lambda x: question_options_tab2[x]
    )
    
    if selected_feature:
        # Group by feature value and calculate mean probability
        risk_by_feature = pd.DataFrame({
            'FeatureValue': X_full[selected_feature],
            'RiskProbability': initial_probs * 100
        })
        
        # Filter valid values (-1 is missing/NA)
        valid_data = risk_by_feature[risk_by_feature['FeatureValue'] >= 0]
        
        fig_feat, ax_feat = plt.subplots(figsize=(10, 5))
        valid_data.groupby('FeatureValue')['RiskProbability'].mean().plot(kind='bar', color='#ff7f0e', ax=ax_feat, edgecolor='black', alpha=0.8)
        ax_feat.set_xlabel("Question Score Level (1=Lowest/Strongly Disagree, 5=Highest/Strongly Agree)")
        ax_feat.set_ylabel("Average Turnover Risk (%)")
        ax_feat.set_title(f"Turnover Risk by Score Level: {selected_feature}")
        plt.xticks(rotation=0)
        st.pyplot(fig_feat)

# Tab 3: Policy Simulation
with tab3:
    st.header("Policy Simulation (What-If Analysis)")
    st.markdown("Simulate improvements in the organizational environment or policies to observe the impact on aggregate turnover risk figures.")
    
    # Extract top 5 features to use as defaults
    importance = uncalib_model.feature_importances_
    indices = np.argsort(importance)[::-1]
    top_5_cols = X_sample.columns[indices][:5].tolist()
    
    # Extract all questions (columns starting with Q)
    question_cols = [col for col in X_sample.columns if col.startswith('Q')]
    question_options = {col: cb_dict.get(col, col) for col in question_cols}

    st.subheader("Select Policy Variables")
    st.markdown("""
Suppose HR can implement policies to increase the satisfaction level of **all employees in the organization** 
from the baseline database. Please use the sliders to add satisfaction scores:

*   **+0:** No additional policies (maintain original FEVS poll satisfaction levels)
*   **+1:** Positive policy impact (satisfaction of all employees increases by 1 level)
*   **+2:** Highly positive policy impact (satisfaction jumps by 2 levels)
*   **+3 to +4:** Macro-level policy reform (satisfaction surges by 3-4 levels)
*   **+5:** Ideal transformation (employees who previously voted 1 automatically switch to a full score of 5)

*(Note: The system will automatically cap the satisfaction score at a maximum of 5 after the addition)*
""")
    
    # Default selection based on Top 5 features: Q94, Q68, Q69, Q43, Q90
    default_sim_cols = [q for q in ['Q94', 'Q68', 'Q69', 'Q43', 'Q90'] if q in question_cols]

    selected_cols = st.multiselect(
        "🔎 Search and select the question topics you are interested in developing:",
        options=question_cols,
        default=default_sim_cols,
        format_func=lambda x: question_options[x]
    )
    
    slider_vals = {}
    if not selected_cols:
        st.warning("⚠️ Please select at least 1 question from the menu above to experiment with adjusting scores.")
    else:
        st.write("---")
        st.write("**Select the expected policy outcome level you want to simulate for each item:**")
        for col in selected_cols:
            label = cb_dict.get(col, col)
            slider_vals[col] = st.slider(f"🌟 {label}", min_value=0, max_value=5, value=0, step=1)
        
    if st.button("🚀 Run Simulation Model", type="primary"):
        # Copy data for simulation from FULL data
        X_simulated = X_full.copy()
        
        # Apply policies
        for col, diff in slider_vals.items():
            if diff > 0:
                # Add and cap at the maximum of the existing column (typically Q1-Q99 are capped at 5)
                max_val = 5 if 'Q' in col else X_simulated[col].max()
                X_simulated[col] = np.clip(X_simulated[col] + diff, -1, max_val)
                
        # Feed data back into the model
        new_probs = model.predict_proba(X_simulated)[:, 1]
        new_avg_risk = np.mean(new_probs) * 100
        new_high_risk_pct = (np.sum(new_probs > 0.5) / len(new_probs)) * 100
        
        st.divider()
        st.subheader("Simulation Results")
        
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            delta_risk = new_avg_risk - avg_risk
            st.metric("Probability of Employee Turnover (Aggregate)", f"{new_avg_risk:.2f}%", f"{delta_risk:.2f}%", delta_color="inverse")
        with sim_col2:
            delta_pct = new_high_risk_pct - high_risk_pct
            st.metric("Proportion of High-Risk Employees", f"{new_high_risk_pct:.2f}%", f"{delta_pct:.2f}%", delta_color="inverse")
            
        st.success("🎉 If these simulated policies are successfully implemented and lead to improved employee survey responses as shown, it will help sustainably reduce employee turnover rates.")
        
        st.markdown("### 📊 Compare Risk Distribution of All Personnel (Before vs After)")
        fig_sim, ax_sim = plt.subplots(figsize=(10, 4))
        ax_sim.hist(initial_probs, bins=50, color='gray', alpha=0.5, edgecolor='black', label=f'Original Mean: {avg_risk:.2f}%')
        ax_sim.hist(new_probs, bins=50, color='#2ca02c', alpha=0.7, edgecolor='black', label=f'Simulated Mean: {new_avg_risk:.2f}%')
        ax_sim.axvline(np.mean(initial_probs), color='gray', linestyle='dashed', linewidth=2)
        ax_sim.axvline(np.mean(new_probs), color='green', linestyle='dashed', linewidth=2)
        ax_sim.set_xlabel("Probability of Turnover Intention")
        ax_sim.set_ylabel("Number of Employees (per risk bin)")
        ax_sim.legend()
        st.pyplot(fig_sim)

# Tab 4: Questionnaire Directory
with tab4:
    st.header("📚 Questionnaire Directory")
    st.markdown("Explore the FEVS (Federal Employee Viewpoint Survey) questions, sections, and indices used in the turnover intention analysis model.")
    
    # Extract unique questions
    unique_qs = codebook[['VARIABLE', 'ITEM TEXT', 'SURVEY SECTION', 'INDEX']].drop_duplicates().reset_index(drop=True)
    
    # Search controls
    col_search, col_filter = st.columns([2, 1])
    with col_search:
        search_query = st.text_input("🔍 Search questions by variable name or keyword (e.g. Q61, satisfaction, supervisor):", "")
    with col_filter:
        sections = ["All"] + sorted(list(unique_qs['SURVEY SECTION'].dropna().unique()))
        selected_section = st.selectbox("📂 Filter by Survey Section:", sections)
        
    # Apply filters
    filtered_qs = unique_qs.copy()
    if search_query:
        filtered_qs = filtered_qs[
            filtered_qs['VARIABLE'].str.contains(search_query, case=False, na=False) |
            filtered_qs['ITEM TEXT'].str.contains(search_query, case=False, na=False)
        ]
    if selected_section != "All":
        filtered_qs = filtered_qs[filtered_qs['SURVEY SECTION'] == selected_section]
        
    st.markdown(f"Showing **{len(filtered_qs)}** matching questions:")
    st.dataframe(
        filtered_qs, 
        use_container_width=True, 
        column_config={
            "VARIABLE": st.column_config.TextColumn("Variable", width="small"),
            "ITEM TEXT": st.column_config.TextColumn("Question text"),
            "SURVEY SECTION": st.column_config.TextColumn("Survey Section", width="medium"),
            "INDEX": st.column_config.TextColumn("Index Reference", width="medium"),
        },
        hide_index=True
    )
