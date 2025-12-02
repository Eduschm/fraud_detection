from app.main import predict, status, get_samples
import streamlit as st
import json


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
        /* Sidebar styles */
        .stSidebar {
            background-color: var(--secondary-bg) !important;
            color: var(--text-color) !important;
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
    page = st.radio("Select Option", ["Dashboard", "Make Prediction"])

if page == "Dashboard":
    st.header("System Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Status", "🟢 Running")
    with col2:
        st.metric("Model", "Active")
    with col3:
        st.metric("API", "Ready")

elif page == "Make Prediction":
    st.header("Fraud Detection Prediction")
    if st.button("Load Sample"):
        sample = get_samples().reset_index(drop=True)
        st.success("Sample Loaded!")
        st.dataframe(sample, use_container_width=False)
        
        

# run app_instance.app.run()

# pip freeze only required
