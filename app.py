from app.main import FraudDetectionApp
import streamlit as st
import json

app_instance = FraudDetectionApp()

# Theme
def set_custom_css():
    st.markdown(
        """
        <style>
        :root {
            --primary-color: #507C7C;
            --bg-color: #709584;
            --secondary-bg: #96C4BB;
            --text-color: #A2D7D6;
        }
        .stApp {
            background-color: var(--bg-color);
            color: var(--text-color);
        }
        .css-1d391kg { 
            background-color: var(--secondary-bg) !important;
        }
        .stButton>button {
            background-color: var(--primary-color) !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

set_custom_css()

st.title("🔍 Fraud Detection System")
st.markdown("---")

# Sidebar for navigation
with st.sidebar:
    st.header("Menu")
    page = st.radio("Select Option", ["Dashboard", "Make Prediction", "View Samples"])

if page == "Dashboard":
    st.header("System Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Status", "🟢 Running")
    with col2:
        st.metric("Model", "Active")
    with col3:
        st.metric("API", "Ready")

elif page == "View Samples":
    st.header("Sample Data")
    if st.button("Generate Sample"):
        samples = app_instance.sample_data.sample(n=5)
        st.success("Samples Generated!")
        st.dataframe(samples, use_container_width=True)

elif page == "Make Prediction":
    st.header("Fraud Detection Prediction")
    
    # Get sample data for reference
    sample_cols = st.columns(2)
    with sample_cols[0]:
        if st.button("Load Sample"):
            sample = app_instance.sample_data.sample(n=1).iloc[0]
            st.json(sample.to_dict())
    
    # Input section
    with st.form("prediction_form"):
        st.write("Enter transaction details:")
        transaction_data = st.text_area("Transaction Data (JSON format)", height=150)
        submitted = st.form_submit_button("🔍 Predict")
        
        if submitted:
            with st.spinner("Analyzing transaction..."):
                try:
                    X = json.loads(transaction_data)
                    result = app_instance.predict(X)
                    st.success("Prediction Complete!")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Risk Level", result.get("risk", "N/A"))
                    with col2:
                        st.metric("Confidence", f"{result.get('confidence', 0):.2%}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# run app_instance.app.run()

# pip freeze only required
