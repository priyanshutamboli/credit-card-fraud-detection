import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('fraud_detection_rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# Title
st.title("🔒 Credit Card Fraud Detector")
st.write("Upload transaction data to detect fraud!")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {len(df)} transactions")
    
    # Show original data in collapsible section
    with st.expander("👁️ View uploaded data"):
        st.dataframe(df.head())
    
    # Preprocess
    df_processed = df.copy()
    df_processed[['Amount', 'Time']] = scaler.transform(df_processed[['Amount', 'Time']])
    
    # Prepare features
    feature_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    X = df_processed[feature_columns]
    
    # Predict
    with st.spinner("🔍 Analyzing transactions..."):
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
    
    # Add results to dataframe
    df['Prediction'] = ['🚨 FRAUD' if p == 1 else '✅ Legitimate' for p in predictions]
    df['Fraud_Probability_%'] = (probabilities * 100).round(2)
    
    # Summary
    st.subheader("📊 Summary")
    col1, col2, col3 = st.columns(3)
    
    total = len(df)
    fraud_count = (predictions == 1).sum()
    legit_count = total - fraud_count
    
    with col1:
        st.metric("Total Transactions", f"{total:,}")
    
    with col2:
        st.metric("🚨 Frauds Detected", fraud_count)
    
    with col3:
        st.metric("✅ Legitimate", legit_count)
    
    # Results in tabs
    st.subheader("🔍 Detailed Results")
    tab1, tab2 = st.tabs(["📋 All Transactions", "🚨 Fraud Only"])
    
    with tab1:
        st.dataframe(
            df[['Amount', 'Prediction', 'Fraud_Probability_%']],
            use_container_width=True
        )
    
    with tab2:
        fraud_df = df[df['Prediction'] == 'FRAUD']
        if len(fraud_df) > 0:
            st.dataframe(
                fraud_df[['Amount', 'Prediction', 'Fraud_Probability_%']].sort_values(
                    'Fraud_Probability_%',
                    ascending=False
                ),
                use_container_width=True
            )
        else:
            st.success("✅ No fraud detected!")
    
    # Download
    st.subheader("Download Results")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Results (CSV)",
        data=csv,
        file_name="fraud_detection_results.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a CSV file to begin")

# Footer
st.divider()
st.caption("Built with Streamlit | Random Forest Model | F1 Score: 85.6%")
