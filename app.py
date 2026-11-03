import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(page_title='Customer Segmentation', page_icon='👥', layout='wide')

# --------------------------------------------------
# Load model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load('customer_segmentation_model.pkl')

bundle = load_model()
pipeline = bundle['pipeline']
cluster_mapping = bundle['cluster_mapping']
num_features = bundle['num_features']
log_features = bundle['log_features']
cat_cols = bundle['cat_cols']
discrete_num_cols = bundle['discrete_num_cols']
raw_log_mapping = bundle['raw_log_mapping']
category_values = bundle['category_values']
numeric_defaults = bundle['numeric_defaults']
model_feature_columns = bundle['model_feature_columns']

# --------------------------------------------------
# Application title
# --------------------------------------------------

st.title('Customer Segmentation Predictor')
st.write('Enter the customer information below to predict the customer segment using the PCA + K-Means model.')

# --------------------------------------------------
# Input form
# --------------------------------------------------

with st.form('customer_input_form'):
    st.subheader('Customer Details')
    inputs = {}
    numerical_input_cols = num_features + discrete_num_cols + list(raw_log_mapping.keys())

    # Prevent accidental duplicate inputs
    numerical_input_cols = list(dict.fromkeys(numerical_input_cols))
    col1, col2, col3 = st.columns(3)
    for index, feature in enumerate(numerical_input_cols):
        target_column = [col1, col2, col3][index % 3]
        default_value = float(numeric_defaults.get(feature, 0.0))

        with target_column:
            if feature in discrete_num_cols:
                inputs[feature] = st.number_input(label=feature.replace('_', ' ').title(), min_value=0, 
                                                  value=max(0, int(round(default_value))), step=1)
            else:
                inputs[feature] = st.number_input(label=feature.replace('_', ' ').title(), min_value=0.0, 
                                                  value=max(0.0, default_value), step=0.01, format='%.2f')
    st.subheader('Customer Preferences')
    cat_col1, cat_col2 = st.columns(2)
    for index, feature in enumerate(cat_cols):
        target_column = (cat_col1 if index % 2 == 0 else cat_col2)
        options = category_values[feature]
        with target_column:
            inputs[feature] = st.selectbox(label=feature.replace('_', ' ').title(), options=options)
    
    submitted = st.form_submit_button('Predict Customer Segment', type='primary')

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if submitted:
    input_data = {}
    for feature in num_features:
        input_data[feature] = inputs[feature]
    for feature in discrete_num_cols:
        input_data[feature] = inputs[feature]
    for raw_feature, transformed_feature in raw_log_mapping.items():
        raw_value = inputs[raw_feature]
        if raw_value < 0:
            st.error(f'{raw_feature} cannot be negative.')
            st.stop()
        input_data[transformed_feature] = np.log1p(raw_value)
    for feature in cat_cols:
        input_data[feature] = inputs[feature]
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=model_feature_columns)
    if input_df.isna().any().any():
        missing_columns = input_df.columns[input_df.isna().any()].tolist()
        st.error('Missing model inputs: '+ ', '.join(missing_columns))
        st.stop()

    # Predict
    cluster_number = int(pipeline.predict(input_df)[0])
    customer_segment = cluster_mapping.get(cluster_number, 'Unknown Segment')

    # Display prediction
    st.success(f'Predicted Customer Segment: {customer_segment}')
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric(label='Cluster Number', value=cluster_number)
    with metric_col2:
        st.metric(label='Customer Segment', value=customer_segment)