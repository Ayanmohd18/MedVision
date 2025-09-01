import streamlit as st
import requests
import json
from PIL import Image
import io

# Page config
st.set_page_config(
    page_title="AI Radiology Report Generator",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("🏥 AI Radiology Report Generator")
st.markdown("Upload a chest X-ray image to generate an automated radiology report")

# Sidebar for patient information
st.sidebar.header("Patient Information")
patient_age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=45)
patient_gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
view_position = st.sidebar.selectbox("View Position", ["PA", "AP", "Lateral"])
clinical_history = st.sidebar.text_area("Clinical History", "No significant medical history")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 Upload Chest X-Ray")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a chest X-ray image in PNG, JPG, or JPEG format"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-Ray", use_container_width=True)
        
        # Analyze button
        if st.button("🔍 Analyze X-Ray", type="primary"):
            with st.spinner("Analyzing chest X-ray..."):
                try:
                    # Prepare files for API request
                    files = {"file": uploaded_file.getvalue()}
                    data = {
                        "patient_age": patient_age,
                        "patient_gender": patient_gender,
                        "view_position": view_position,
                        "clinical_history": clinical_history
                    }
                    
                    # Make API request
                    response = requests.post(
                        "http://127.0.0.1:8003/analyze-xray",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                        data=data
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state['analysis_result'] = result
                        st.success("Analysis completed successfully!")
                    else:
                        st.error(f"Analysis failed: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Make sure the API server is running on port 8003.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

with col2:
    st.header("📋 Analysis Results")
    
    if 'analysis_result' in st.session_state:
        result = st.session_state['analysis_result']
        
        # Display findings
        st.subheader("🔍 Detected Findings")
        findings = result.get('findings', [])
        
        for finding in findings:
            if finding['probability'] > 0.3:
                confidence = "🔴 High" if finding['probability'] > 0.7 else "🟡 Medium"
                st.write(f"**{finding['pathology']}**: {finding['probability']:.1%} ({confidence})")
        
        # Display report
        if 'report' in result:
            report = result['report']
            
            st.subheader("📝 Radiology Report")
            
            with st.expander("🔍 Findings", expanded=True):
                st.write(report.get('findings', 'No findings available'))
            
            with st.expander("💡 Impression", expanded=True):
                st.write(report.get('impression', 'No impression available'))
            
            with st.expander("📋 Recommendations", expanded=True):
                st.write(report.get('recommendations', 'No recommendations available'))
        
        # Quality metrics
        if 'quality_metrics' in result:
            quality = result['quality_metrics']
            st.subheader("📊 Image Quality")
            
            col_q1, col_q2 = st.columns(2)
            with col_q1:
                st.metric("Image Valid", "✅ Yes" if quality.get('is_valid') else "❌ No")
            with col_q2:
                st.metric("Mean Intensity", f"{quality.get('mean_intensity', 0):.3f}")
        
        # Confidence summary
        if 'confidence_summary' in result:
            confidence = result['confidence_summary']
            st.subheader("🎯 Confidence Summary")
            
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.metric("Max Confidence", f"{confidence.get('max_confidence', 0):.1%}")
            with col_c2:
                st.metric("Significant Findings", confidence.get('significant_findings', 0))
    
    else:
        st.info("👆 Upload an X-ray image and click 'Analyze' to see results")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>⚠️ <strong>Disclaimer:</strong> This is a research tool. Not for clinical diagnosis.</p>
        <p>🔬 AI Radiology System v1.0</p>
    </div>
    """,
    unsafe_allow_html=True
)