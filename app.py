import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from datetime import datetime, timedelta
import time

# --- CONFIGURATION ---
# API Key configuration for deployment
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("⚠️ Please configure GOOGLE_API_KEY in Streamlit secrets")
    st.info("For local development, set your API key in .streamlit/secrets.toml")
    st.stop()

genai.configure(api_key=API_KEY)

# --- API USAGE CONTROL ---
# Daily usage limits (adjust these numbers as needed)
MAX_DAILY_REQUESTS = 50  # 每日最多分析次數
MAX_HOURLY_REQUESTS = 10  # 每小時最多分析次數

def check_usage_limits():
    """Check if user has exceeded usage limits"""
    # Initialize session state for usage tracking
    if 'usage_count' not in st.session_state:
        st.session_state.usage_count = 0
    if 'last_reset' not in st.session_state:
        st.session_state.last_reset = datetime.now()
    if 'hourly_count' not in st.session_state:
        st.session_state.hourly_count = 0
    if 'last_hour_reset' not in st.session_state:
        st.session_state.last_hour_reset = datetime.now()
    
    # Reset daily counter
    if datetime.now() - st.session_state.last_reset > timedelta(days=1):
        st.session_state.usage_count = 0
        st.session_state.last_reset = datetime.now()
    
    # Reset hourly counter
    if datetime.now() - st.session_state.last_hour_reset > timedelta(hours=1):
        st.session_state.hourly_count = 0
        st.session_state.last_hour_reset = datetime.now()
    
    # Check limits
    if st.session_state.usage_count >= MAX_DAILY_REQUESTS:
        st.error("🚫 Daily usage limit reached (50 analyses). Please try again tomorrow.")
        st.info("This limit helps prevent unexpected API charges.")
        return False
    
    if st.session_state.hourly_count >= MAX_HOURLY_REQUESTS:
        st.error("⏰ Hourly usage limit reached (10 analyses). Please wait an hour.")
        return False
    
    return True

def increment_usage():
    """Increment usage counters"""
    st.session_state.usage_count += 1
    st.session_state.hourly_count += 1

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
    # Check usage limits before API call
    if not check_usage_limits():
        return None
    
    # Add rate limiting (prevent spam)
    time.sleep(1)
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""
        Act as a Senior Career Coach and Resume Writer. 
        Analyze the following resume text and provide a professional review.
        
        Resume Text:
        {resume_text[:3000]}  # Limit input length to control costs
        
        Output the response in Markdown with these headers:
        1. **🏆 Professional Summary**: Rate the summary section (if it exists) and suggest a stronger 2-sentence version.
        2. **💪 Top Strengths**: identifying the top 3-5 distinct skills or experiences that stand out.
        3. **🛑 Areas for Improvement**: Identify weak verbs, passive language, or formatting issues.
        4. **✨ Recommended Roles**: Based on this resume, what are 3 job titles this candidate is best suited for?
        """
        response = model.generate_content(prompt)
        increment_usage()  # Only count successful requests
        return response.text
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def analyze_resume_vs_jd(resume_text, jd_text):
    """Mode 2: Comparison with JD."""
    # Check usage limits before API call
    if not check_usage_limits():
        return None
    
    # Add rate limiting (prevent spam)
    time.sleep(1)
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""
        Act as an Application Tracking System (ATS) and Technical Recruiter.
        Compare the Resume against the Job Description.
        
        Resume: {resume_text[:2000]}  # Limit input length
        Job Description: {jd_text[:1000]}  # Limit input length
        
        Output in Markdown:
        1. **🚫 Missing Critical Skills**: List technical skills/tools in the JD that are NOT in the resume.
        2. **📉 Gap Analysis**: Briefly explain where the candidate's experience falls short of the requirements.
        3. **✍️ Bullet Point Rewrite**: Pick one existing bullet point from the resume and rewrite it to specifically target this job description using keywords from the JD.
        4. **⚖️ Final Verdict**: "High Match", "Medium Match", or "Low Match" with a 1-sentence reason.
        """
        response = model.generate_content(prompt)
        increment_usage()  # Only count successful requests
        return response.text
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

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
    
    # Usage statistics
    st.subheader("📊 Usage Today")
    if 'usage_count' in st.session_state:
        remaining = MAX_DAILY_REQUESTS - st.session_state.usage_count
        st.metric("Analyses Remaining", remaining)
        st.progress(st.session_state.usage_count / MAX_DAILY_REQUESTS)
    else:
        st.metric("Analyses Remaining", MAX_DAILY_REQUESTS)

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
                if analysis:  # Only show results if analysis was successful
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
                    # 1. Math Score (doesn't use API)
                    match_score = calculate_match_score(resume_text, jd_input)
                    
                    # 2. AI Analysis (uses API)
                    ai_analysis = analyze_resume_vs_jd(resume_text, jd_input)
                    
                    # Display Results
                    st.divider()
                    st.subheader("📊 Analysis Results")
                    
                    # Always show math score (free calculation)
                    score_color = "red"
                    if match_score > 75: score_color = "green"
                    elif match_score > 50: score_color = "orange"
                    
                    st.markdown(f"### ATS Match Score: :{score_color}[{match_score}%]")
                    st.progress(int(match_score))
                    
                    # Only show AI analysis if successful
                    if ai_analysis:
                        st.markdown("---")
                        st.markdown(ai_analysis)
        else:
            st.warning("Please upload both a Resume and a Job Description.")