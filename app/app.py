from main import FraudDetectionApp
import app.app as st

app_instance = FraudDetectionApp()

# Streamlit config
st.set_page_config(page_title="Fraud Detection API", layout="centered")
# Theme
def set_custom_css():
    st.markdown(
        """
        <style>
        :root {
            --primary-color: #1abc9c;
            --bg-color: #0e1117;
            --secondary-bg: #111318;
            --text-color: #e6eef8;
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


# Streamlit interface


st.title("Fraud Detection API Status")
st.write("API is running and ready to accept requests.")

# run app_instance.app.run()

# pip freeze only required
