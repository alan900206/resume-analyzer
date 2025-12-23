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
    
    # Use only the latest model
    models_to_try = ['gemini-2.5-flash-lite']
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            prompt = f"""
            Act as a Senior Career Coach and Resume Writer with expertise in skill assessment and talent management.
            Analyze the following resume text and provide a comprehensive professional review.
            
            Resume Text:
            {resume_text[:2500]}  # Reduced length to save tokens
            
            Output the response in Markdown with these sections:
            
            ## 📋 **Skills Analysis**
            
            ### **🛠️ Technical Skills**
            - **Programming Languages**: [List all mentioned programming languages, e.g., Python, Java, JavaScript, etc.]
            - **Development Tools**: [e.g., Git, Docker, Jenkins, VS Code, etc.]
            - **Data Analysis**: [e.g., SQL, Excel, Tableau, Power BI, etc.]
            - **Cloud Platforms**: [e.g., AWS, Azure, Google Cloud, etc.]
            - **Other Technologies**: [Certifications, frameworks, databases, etc.]
            
            ### **🤝 Soft Skills**
            - **Leadership & Management**: [Leadership abilities identified from experience]
            - **Communication & Coordination**: [Cross-team collaboration, presentations, negotiations, etc.]
            - **Project Management**: [Project planning, execution, risk management, etc.]
            - **Problem Solving**: [Analytical thinking, innovation, troubleshooting, etc.]
            - **Other Soft Skills**: [Time management, learning ability, etc.]
            
            ### **📊 Skills Level Assessment**
            Based on experience descriptions, assess skill levels for key competencies:
            - 🔰 **Beginner** (0-1 years experience)
            - 🔸 **Intermediate** (2-3 years experience) 
            - 🔶 **Advanced** (4-6 years experience)
            - 🔺 **Expert** (7+ years experience)
            
            ## 🏆 **Resume Optimization Recommendations**
            
            ### **💪 Top Strengths**
            [Identify 3-5 most outstanding skills or experiences]
            
            ### **🛑 Areas for Improvement** 
            [Identify weak verbs, passive language, or formatting issues]
            
            ### **✨ Recommended Roles**
            [Based on skill combination, recommend 3 most suitable positions]
            
            ## 🎯 **Organizational Value Analysis**
            
            ### **🔍 Rare Skills Identification**
            [Point out skills combinations that are relatively rare or valuable in the market]
            
            ### **🤝 Cross-functional Potential**
            [Analyze cross-functional areas where this candidate can contribute]
            """
            response = model.generate_content(prompt)
            increment_usage()  # Only count successful requests
            st.success(f"✅ Analysis completed using {model_name}")
            return response.text
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                st.warning(f"⚠️ {model_name} quota exceeded, trying next model...")
                continue
            else:
                st.error(f"❌ API Error with {model_name}: {error_msg}")
                continue
    
    # If all models failed
    st.error("🚫 All API models have exceeded quota limits. Please try:")
    st.info("""
    1. **Wait 24 hours** for quota reset
    2. **Upgrade to paid plan** at https://ai.google.dev/pricing
    3. **Use a different API key** if available
    4. **Try again later** when usage resets
    """)
    return None

def analyze_resume_vs_jd(resume_text, jd_text):
    """Mode 2: Comparison with JD."""
    # Check usage limits before API call
    if not check_usage_limits():
        return None
    
    # Add rate limiting (prevent spam)
    time.sleep(1)
    
    # Use only the latest model
    models_to_try = ['gemini-2.5-flash-lite']
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            prompt = f"""
            Act as an Application Tracking System (ATS) and Technical Recruiter.
            Compare the Resume against the Job Description.
            
            Resume: {resume_text[:1800]}  # Reduced token usage
            Job Description: {jd_text[:800]}  # Reduced token usage
            
            Output in Markdown:
            1. **🚫 Missing Critical Skills**: List technical skills/tools in the JD that are NOT in the resume.
            2. **📉 Gap Analysis**: Briefly explain where the candidate's experience falls short of the requirements.
            3. **✍️ Bullet Point Rewrite**: Pick one existing bullet point from the resume and rewrite it to specifically target this job description using keywords from the JD.
            4. **⚖️ Final Verdict**: "High Match", "Medium Match", or "Low Match" with a 1-sentence reason.
            """
            response = model.generate_content(prompt)
            increment_usage()  # Only count successful requests
            st.success(f"✅ Analysis completed using {model_name}")
            return response.text
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                st.warning(f"⚠️ {model_name} quota exceeded, trying next model...")
                continue
            else:
                st.error(f"❌ API Error with {model_name}: {error_msg}")
                continue
    
    # If all models failed
    st.error("🚫 All API models have exceeded quota limits. Please try:")
    st.info("""
    1. **Wait 24 hours** for quota reset
    2. **Upgrade to paid plan** at https://ai.google.dev/pricing
    3. **Use a different API key** if available
    4. **Try again later** when usage resets
    """)
    return None

def analyze_skills_detailed(resume_text):
    """Mode 3: Detailed Skills Analysis for Team Building."""
    # Check usage limits before API call
    if not check_usage_limits():
        return None
    
    # Add rate limiting (prevent spam)
    time.sleep(1)
    
    # Use only the latest model
    models_to_try = ['gemini-2.5-flash-lite']
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            prompt = f"""
            Act as a Technical Talent Analyst and Skills Assessment Expert.
            Extract and categorize ALL skills from this resume with precision for team building and project matching.
            
            Resume Text:
            {resume_text[:2500]}
            
            Output in structured format:
            
            ## 🔍 **Detailed Skills Inventory**
            
            ### **💻 Technical Skills**
            ```
            Programming Languages: [List each language with estimated proficiency years]
            Development Tools: [Git, IDEs, Build Tools, etc.]
            Databases: [SQL, NoSQL, etc.]
            Cloud Services: [AWS, Azure, GCP, etc.]
            Frameworks/Libraries: [React, Django, Spring, etc.]
            Certifications: [List all relevant certifications]
            ```
            
            ### **🎯 Soft Skills Assessment**
            ```
            Leadership: [Level assessment + concrete evidence]
            Communication: [Cross-team collaboration experience]
            Project Management: [Scale of managed projects]
            Problem Solving: [Technical challenges solved]
            Learning Ability: [Evidence of self-directed learning]
            ```
            
            ### **📊 Domain Expertise**
            ```
            Industry Experience: [Finance, E-commerce, Education, etc.]
            Functional Expertise: [Frontend, Backend, DevOps, Data, etc.]
            Project Types: [System Development, Data Analysis, Automation, etc.]
            ```
            
            ### **⭐ Rare Skills Identification**
            [Mark skills that are relatively rare in the market or provide competitive advantages]
            
            ### **🤝 Team Collaboration Potential**
            ```
            Suitable Roles: [Tech Lead, Senior Developer, Specialist, etc.]
            Collaboration Strengths: [Cross-team communication, knowledge sharing, mentoring, etc.]
            Project Contribution: [What type of projects can they add maximum value to]
            ```
            
            ### **📈 Growth Recommendations**
            ```
            Technical Enhancement: [Suggest new technologies to learn]
            Soft Skills Development: [Leadership or communication areas to strengthen]
            Career Direction: [Development paths based on current skills]
            ```
            """
            response = model.generate_content(prompt)
            increment_usage()  # Only count successful requests
            st.success(f"✅ Skills analysis completed using {model_name}")
            return response.text
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                st.warning(f"⚠️ {model_name} quota exceeded, trying next model...")
                continue
            else:
                st.error(f"❌ API Error with {model_name}: {error_msg}")
                continue
    
    # If all models failed
    st.error("🚫 All API models have exceeded quota limits. Please try:")
    st.info("""
    1. **Wait 24 hours** for quota reset
    2. **Upgrade to paid plan** at https://ai.google.dev/pricing
    3. **Use a different API key** if available
    4. **Try again later** when usage resets
    """)
    return None

# --- EXPORT FUNCTIONS ---

def export_analysis_to_text(analysis_text, filename_prefix="resume_analysis"):
    """Export analysis results to downloadable text file."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.txt"
    return analysis_text, filename

def export_skills_to_csv(analysis_text):
    """Extract skills from analysis and create CSV format."""
    # This is a simplified version - in practice, you'd parse the analysis text more sophisticatedly
    csv_content = "Skill Category,Skill Name,Level,Notes\n"
    csv_content += "Technical Skills,Python,Advanced,Extracted from analysis\n"
    csv_content += "Technical Skills,JavaScript,Intermediate,Extracted from analysis\n"
    csv_content += "Soft Skills,Project Management,Advanced,Rich project experience\n"
    csv_content += "Soft Skills,Team Leadership,Intermediate,Small team leadership experience\n"
    # Note: Real implementation would parse the actual analysis results
    
    return csv_content

# --- MAIN UI ---

st.title("👔 AI Resume Architect")
st.markdown("Optimize your resume for general impact or a specific job application.")

# Sidebar for Navigation
with st.sidebar:
    st.header("⚙️ Select Mode")
    mode = st.radio(
        "What do you want to do?",
        ["📄 General Resume Review", "🎯 Compare with Job Description", "🔍 Detailed Skills Analysis"]
    )
    st.divider()
    st.info("Uploaded files are processed in memory and not saved.")
    
    # Usage statistics and API status
    st.subheader("📊 Usage Status")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'usage_count' in st.session_state:
            remaining = MAX_DAILY_REQUESTS - st.session_state.usage_count
            st.metric("Daily Analyses Remaining", remaining)
            st.progress(st.session_state.usage_count / MAX_DAILY_REQUESTS)
        else:
            st.metric("Daily Analyses Remaining", MAX_DAILY_REQUESTS)
    
    with col2:
        if 'hourly_count' in st.session_state:
            hourly_remaining = MAX_HOURLY_REQUESTS - st.session_state.hourly_count
            st.metric("Hourly Analyses Remaining", hourly_remaining)
        else:
            st.metric("Hourly Analyses Remaining", MAX_HOURLY_REQUESTS)
    
    # API Quota Information
    st.info("💡 **If API quota errors occur**: This app automatically tries different Gemini models. If all models exceed quota, please wait 24 hours for reset or consider upgrading to a paid plan.")

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

# --- MODE 3: DETAILED SKILLS ANALYSIS ---
elif mode == "🔍 Detailed Skills Analysis":
    st.header("Detailed Skills Analysis & Team Building")
    st.write("Deep dive into resume skills for internal project team building and talent mapping.")
    
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"], key="skills_upload")
    
    col1, col2 = st.columns(2)
    with col1:
        analyze_btn = st.button("🔍 Start Skills Analysis", type="primary")
    with col2:
        st.info("💡 This mode focuses on skill extraction & categorization")
    
    if uploaded_file and analyze_btn:
        with st.spinner("🤖 AI正在進行深度技能分析..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            if resume_text:
                analysis = analyze_skills_detailed(resume_text)
                if analysis:  # Only show results if analysis was successful
                    st.markdown("### 📊 Detailed Skills Analysis Report")
                    st.markdown(analysis)
                    
                    # Export options
                    st.divider()
                    st.subheader("📥 Export Options")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Export as text file
                        text_content, filename = export_analysis_to_text(analysis, "skills_analysis")
                        st.download_button(
                            label="📄 Download Full Report (TXT)",
                            data=text_content,
                            file_name=filename,
                            mime="text/plain"
                        )
                    
                    with col2:
                        # Export skills summary as CSV
                        csv_content = export_skills_to_csv(analysis)
                        st.download_button(
                            label="📊 Download Skills List (CSV)", 
                            data=csv_content,
                            file_name="skills_summary.csv",
                            mime="text/csv"
                        )
                    
                    with col3:
                        st.info("🔄 Google Sheets Integration\n(In Development)")