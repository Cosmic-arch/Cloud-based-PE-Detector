import streamlit as st
import boto3
import json
import tempfile
import os
import numpy as np
import time
import joblib
import ember
import matplotlib.pyplot as plt


# Page setup
st.set_page_config(
    page_title="Malware Detection System",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ PE Malware Detection System")
st.markdown("This system analyzes PE files to determine if they contain malware.")

# Sidebar info
with st.sidebar:
    st.title("About")
    st.info("Upload a Windows PE file to check if it's malware using an ML model deployed on AWS SageMaker.")
    st.title("How it works")
    st.markdown("""
        1. Upload a PE file
        2. Features are extracted locally
        3. Features are sent to a SageMaker endpoint
        4. The model returns a prediction
    """)

# SageMaker endpoint name
ENDPOINT_NAME = "malware-detection-v2-final"  # Change to your actual endpoint

# Feature extraction
def extract_ember_features(pe_path, scaler_path="scaler.joblib"):
    try:
        extractor = ember.PEFeatureExtractor()
        features = extractor.feature_vector(pe_path)
        features = np.array(features, dtype=np.float32).reshape(1, -1)
        scaler = joblib.load(scaler_path)
        features_scaled = scaler.transform(features)
        return features_scaled.reshape(1, 1, -1).astype(np.float32)
    except Exception as e:
        st.error(f"Feature extraction failed: {str(e)}")
        return None

# Call SageMaker endpoint
def query_endpoint(features, endpoint_name):
    client = boto3.client('sagemaker-runtime')
    input_data = {'features': features.tolist()}
    try:
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(input_data)
        )
        result = json.loads(response['Body'].read().decode())
        return result
    except Exception as e:
        st.error(f"Error querying endpoint: {str(e)}")
        return {"prediction": "Error", "probability": 0.5}

# Upload PE file
st.header("Upload a PE File")
uploaded_file = st.file_uploader("Choose a PE file", type=['exe', 'dll', 'sys'])

if uploaded_file is not None:
    st.write("### File Details")
    st.write(f"**Filename:** {uploaded_file.name}")
    st.write(f"**File size:** {uploaded_file.size / 1024:.2f} KB")

    if st.button("Analyze File"):
        progress_bar = st.progress(0)

        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.exe') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        progress_bar.progress(20)
        st.write("Extracting features...")
        features = extract_ember_features(tmp_path)

        if features is None:
            st.stop()

        progress_bar.progress(50)
        st.write("Sending to SageMaker endpoint...")
        start_time = time.time()
        result = query_endpoint(features, ENDPOINT_NAME)
        end_time = time.time()

        progress_bar.progress(90)
        st.write("### Prediction Result")
        st.write(f"**Processing Time:** {end_time - start_time:.2f} seconds")

        if result["prediction"] == "Malware":
            st.error(f"⚠️ MALWARE DETECTED with {result['probability']*100:.2f}% confidence!")
        elif result["prediction"] == "Benign":
            st.success(f"✅ BENIGN FILE with {(1 - result['probability'])*100:.2f}% confidence.")
        else:
            st.warning("Could not determine the file type.")

        with st.expander("Show Technical Details"):
            st.write(f"Raw probability score: {result['probability']:.6f}")
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.barh(['Benign', 'Malware'], [1 - result['probability'], result['probability']], color=['green', 'red'])
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability')
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        progress_bar.progress(100)
        os.unlink(tmp_path)

# Footer
st.markdown("---")
st.caption("DSCI 6015: AI and Cybersecurity - Spring 2025")
st.caption("Created for the Cloud-based PE Malware Detection Project")
