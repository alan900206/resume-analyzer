import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# --- CONFIGURATION ---
# API Key configuration for deployment
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    # For local development, you can uncomment and add your key here
    # API_KEY = "YOUR_API_KEY_HERE"
    st.error("⚠️ Please configure GOOGLE_API_KEY in Streamlit secrets")
    st.stop()

genai.configure(api_key=API_KEY)

# Page Config
st.set_page_config(page_title="AI Resume Expert", layout="wide", page_icon="👔")

# --- FUNCTIONS ---

def extract_text_from_pdf(uploaded_file):
    """Extracts text from PDF."""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def calculate_match_score(resume_text, jd_text):
    """Math-based ATS score."""
    text_list = [resume_text, jd_text]
    cv = TfidfVectorizer()
    count_matrix = cv.fit_transform(text_list)
    match_percentage = cosine_similarity(count_matrix)[0][1] * 100
    return round(match_percentage, 2)

def analyze_resume_general(resume_text):
    """Mode 1: General Analysis without JD."""
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"""
    Act as a Senior Career Coach and Resume Writer. 
    Analyze the following resume text and provide a professional review.
    
    Resume Text:
    {resume_text}
    
    Output the response in Markdown with these headers:
    1. **🏆 Professional Summary**: Rate the summary section (if it exists) and suggest a stronger 2-sentence version.
    2. **💪 Top Strengths**: identifying the top 3-5 distinct skills or experiences that stand out.
    3. **🛑 Areas for Improvement**: Identify weak verbs, passive language, or formatting issues.
    4. **✨ Recommended Roles**: Based on this resume, what are 3 job titles this candidate is best suited for?
    """
    response = model.generate_content(prompt)
    return response.text

def analyze_resume_vs_jd(resume_text, jd_text):
    """Mode 2: Comparison with JD."""
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"""
    Act as an Application Tracking System (ATS) and Technical Recruiter.
    Compare the Resume against the Job Description.
    
    Resume: {resume_text}
    Job Description: {jd_text}
    
    Output in Markdown:
    1. **🚫 Missing Critical Skills**: List technical skills/tools in the JD that are NOT in the resume.
    2. **📉 Gap Analysis**: Briefly explain where the candidate's experience falls short of the requirements.
    3. **✍️ Bullet Point Rewrite**: Pick one existing bullet point from the resume and rewrite it to specifically target this job description using keywords from the JD.
    4. **⚖️ Final Verdict**: "High Match", "Medium Match", or "Low Match" with a 1-sentence reason.
    """
    response = model.generate_content(prompt)
    return response.text

# --- MAIN UI ---

st.title("👔 AI Resume Architect")
st.markdown("Optimize your resume for general impact or a specific job application.")

# Sidebar for Navigation
with st.sidebar:
    st.header("⚙️ Select Mode")
    mode = st.radio(
        "What do you want to do?",
        ["📄 General Resume Review", "🎯 Compare with Job Description"]
    )
    st.divider()
    st.info("Uploaded files are processed in memory and not saved.")

# --- MODE 1: GENERAL REVIEW ---
if mode == "📄 General Resume Review":
    st.header("General Resume Health Check")
    st.write("Upload your resume to get insights on your strengths, weaknesses, and suitable roles.")
    
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    
    if uploaded_file and st.button("Analyze Resume"):
        with st.spinner("🔍 AI is reviewing your profile..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            if resume_text:
                analysis = analyze_resume_general(resume_text)
                st.markdown("### 📝 AI Analysis Report")
                st.markdown(analysis)

# --- MODE 2: COMPARE WITH JD ---
elif mode == "🎯 Compare with Job Description":
    st.header("Job Fit & ATS Scanner")
    st.write("Check how well your resume matches a specific job opening.")
    
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("1. Upload Resume (PDF)", type=["pdf"])
    with col2:
        jd_input = st.text_area("2. Paste Job Description", height=150)
        
    if st.button("Compare Resume"):
        if uploaded_file and jd_input:
            with st.spinner("🤖 Calculating match score and finding gaps..."):
                resume_text = extract_text_from_pdf(uploaded_file)
                
                if resume_text:
                    # 1. Math Score
                    match_score = calculate_match_score(resume_text, jd_input)
                    
                    # 2. AI Analysis
                    ai_analysis = analyze_resume_vs_jd(resume_text, jd_input)
                    
                    # Display Results
                    st.divider()
                    st.subheader("📊 Analysis Results")
                    
                    # Gauge Chart Logic
                    score_color = "red"
                    if match_score > 75: score_color = "green"
                    elif match_score > 50: score_color = "orange"
                    
                    st.markdown(f"### ATS Match Score: :{score_color}[{match_score}%]")
                    st.progress(int(match_score))
                    
                    st.markdown("---")
                    st.markdown(ai_analysis)
        else:
            st.warning("Please upload both a Resume and a Job Description.")