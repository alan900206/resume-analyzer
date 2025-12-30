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

# --- APPIER SOFT SKILLS FRAMEWORK ---
APPIER_SOFT_SKILLS_FRAMEWORK = {
    "Problem Solving & Critical thinking": [
        "Problem Definition & Root Cause",
        "Analytical & Logical Thinking", 
        "Creative & Systems Thinking",
        "Decision-Making & Execution",
        "Adaptive & Collaborative Problem Solving"
    ],
    "Communication & Teamwork": [
        "Active Listening & Empathy",
        "Clear & Impactful Expression",
        "Persuasive & Stakeholder Communication", 
        "Feedback & Constructive Dialogue",
        "Collaboration & Facilitation",
        "Language Proficiency (Cross-Cultural Communication)"
    ],
    "AI Capability": [
        "AI Literacy & Awareness",
        "Applied AI & Data-Driven Problem Solving",
        "Ethical & Responsible AI Practice"
    ],
    "Leadership": [
        "Leading Self - Vision & Direction",
        "Leading Self - Decision-Making Under Uncertainty", 
        "Leading Self - Personal Resilience & Sustainable Performance",
        "Leading Others - People Management & Coaching",
        "Leading Others - Stakeholder Influence & Executive Communication",
        "Leading Others - Change Leadership",
        "Leading Business - Strategic Thinking & Prioritization"
    ],
    "Appier Leader": [
        "Growth & Development",
        "Timely Feedback",
        "Respect & Professionalism",
        "Integrity & Trust", 
        "Genuine Care",
        "Remove Obstacles",
        "Focus & Priorities",
        "Judgment & Expertise"
    ],
    "Appier Core Values": [
        "Direct Communication",
        "Open-Mindedness", 
        "Ambition"
    ]
}

# --- SKILL KEYWORDS MAPPING ---
SKILL_KEYWORDS_MAPPING = {
    "Problem Definition & Root Cause": [
        "問題分析", "根本原因", "根因分析", "5 why", "issue identification", 
        "problem solving", "root cause analysis", "troubleshooting", "問題定義",
        "系統性分析", "問題診斷", "故障排除", "issue resolution"
    ],
    "Analytical & Logical Thinking": [
        "邏輯思考", "分析能力", "analytical", "logical thinking", "data analysis",
        "邏輯推理", "理性分析", "systematic analysis", "critical analysis",
        "數據分析", "quantitative analysis", "邏輯判斷"
    ],
    "Creative & Systems Thinking": [
        "創新思維", "系統思考", "creative", "innovation", "systems thinking",
        "創意", "設計思考", "design thinking", "holistic approach",
        "整體思維", "創造力", "breakthrough thinking", "out of box"
    ],
    "Decision-Making & Execution": [
        "決策", "執行力", "decision making", "execution", "implementation",
        "落地執行", "推動執行", "決策能力", "執行能力", "deliver results",
        "project execution", "follow through", "成果交付"
    ],
    "Adaptive & Collaborative Problem Solving": [
        "適應性", "協作", "collaboration", "adaptability", "flexible",
        "團隊合作", "跨部門", "cross-functional", "agile", "敏捷",
        "彈性", "變通", "teamwork", "partnership"
    ],
    "Active Listening & Empathy": [
        "傾聽", "同理心", "empathy", "active listening", "understanding",
        "換位思考", "情感智慧", "emotional intelligence", "compassion",
        "理解他人", "關懷", "empathetic", "人文關懷"
    ],
    "Clear & Impactful Expression": [
        "表達能力", "簡報", "presentation", "communication", "articulation",
        "清晰表達", "影響力", "說服力", "演講", "public speaking",
        "溝通技巧", "表達技巧", "clear communication"
    ],
    "Persuasive & Stakeholder Communication": [
        "說服", "利害關係人", "stakeholder", "persuasion", "influence",
        "談判", "negotiation", "stakeholder management", "關係建立",
        "影響力溝通", "external communication", "客戶溝通"
    ],
    "Feedback & Constructive Dialogue": [
        "回饋", "建設性對話", "feedback", "constructive", "dialogue",
        "意見交流", "討論", "建議", "改善建議", "coaching",
        "指導", "mentor", "輔導", "建設性溝通"
    ],
    "Collaboration & Facilitation": [
        "協作", "促進", "facilitation", "collaboration", "teamwork",
        "團隊協作", "跨部門合作", "cross-team", "workshop", "會議主持",
        "團隊建設", "consensus building", "協調"
    ],
    "Language Proficiency (Cross-Cultural Communication)": [
        "多語言", "跨文化", "cross-cultural", "international", "global",
        "英文", "中文", "bilingual", "multicultural", "cultural awareness",
        "國際化", "文化敏感度", "語言能力"
    ],
    "AI Literacy & Awareness": [
        "AI", "人工智慧", "machine learning", "AI工具", "ChatGPT", "GPT",
        "AI素養", "AI認知", "artificial intelligence", "ML", "deep learning",
        "AI應用", "AI趨勢", "AI awareness", "AI literacy"
    ],
    "Applied AI & Data-Driven Problem Solving": [
        "數據驅動", "data-driven", "AI應用", "applied AI", "data analysis",
        "數據分析", "AI解決方案", "AI implementation", "automation",
        "自動化", "數據科學", "data science", "AI工具應用"
    ],
    "Ethical & Responsible AI Practice": [
        "AI倫理", "負責任AI", "ethical AI", "responsible AI", "AI ethics",
        "AI風險", "AI安全", "AI governance", "bias", "偏見",
        "公平性", "透明度", "AI責任", "ethical technology"
    ],
    "Leading Self - Vision & Direction": [
        "願景", "方向", "vision", "direction", "goal setting", "目標設定",
        "自我領導", "個人願景", "strategic vision", "long-term thinking",
        "未來規劃", "方向感", "purpose", "使命感"
    ],
    "Leading Self - Decision-Making Under Uncertainty": [
        "不確定性", "uncertainty", "ambiguity", "風險決策", "risk management",
        "模糊情況", "快速決策", "判斷力", "decisiveness", "ambiguous situation",
        "不確定環境", "風險評估", "risk assessment"
    ],
    "Leading Self - Personal Resilience & Sustainable Performance": [
        "韌性", "resilience", "可持續性", "sustainable", "抗壓性",
        "壓力管理", "stress management", "持久力", "endurance", "recovery",
        "自我調節", "work-life balance", "身心健康", "mental health"
    ],
    "Leading Others - People Management & Coaching": [
        "人才管理", "coaching", "輔導", "mentor", "team management",
        "人員發展", "talent development", "績效管理", "performance management",
        "團隊領導", "team leadership", "培養人才", "人才培育"
    ],
    "Leading Others - Stakeholder Influence & Executive Communication": [
        "高層溝通", "executive communication", "影響力", "influence", "stakeholder",
        "利害關係人", "高階主管", "senior management", "board communication",
        "策略溝通", "strategic communication", "executive presence"
    ],
    "Leading Others - Change Leadership": [
        "變革領導", "change leadership", "change management", "transformation",
        "組織變革", "變革管理", "轉型", "digital transformation",
        "文化變革", "cultural change", "innovation leadership"
    ],
    "Leading Business - Strategic Thinking & Prioritization": [
        "策略思維", "strategic thinking", "優先順序", "prioritization", "strategy",
        "商業策略", "business strategy", "策略規劃", "strategic planning",
        "資源配置", "resource allocation", "策略執行"
    ],
    "Growth & Development": [
        "成長", "發展", "growth", "development", "learning", "學習",
        "進步", "improvement", "skill development", "continuous learning",
        "自我提升", "personal development", "professional growth"
    ],
    "Timely Feedback": [
        "及時回饋", "timely feedback", "即時反饋", "real-time feedback",
        "適時指導", "及時溝通", "prompt response", "timely communication",
        "即時回應", "快速反饋", "responsive feedback"
    ],
    "Respect & Professionalism": [
        "尊重", "專業", "respect", "professionalism", "professional conduct",
        "職業操守", "專業態度", "professional behavior", "courtesy",
        "禮貌", "職業素養", "professional ethics", "mutual respect"
    ],
    "Integrity & Trust": [
        "誠信", "信任", "integrity", "trust", "honesty", "trustworthy",
        "可信賴", "誠實", "ethical", "moral", "正直", "品德",
        "職業道德", "professional ethics", "reliability"
    ],
    "Genuine Care": [
        "真心關懷", "genuine care", "caring", "compassion", "empathy",
        "關心", "人文關懷", "人性化", "温暖", "支持", "support",
        "關愛", "體貼", "considerate", "thoughtful"
    ],
    "Remove Obstacles": [
        "排除障礙", "remove obstacles", "problem solving", "barrier removal",
        "清除阻礙", "解決阻礙", "obstacle management", "roadblock removal",
        "困難排除", "解決困難", "enable others", "facilitate progress"
    ],
    "Focus & Priorities": [
        "專注", "優先順序", "focus", "priorities", "concentration",
        "重點", "核心", "key focus", "strategic focus", "priority management",
        "時間管理", "attention management", "goal-oriented"
    ],
    "Judgment & Expertise": [
        "判斷力", "專業知識", "judgment", "expertise", "professional knowledge",
        "專業能力", "經驗", "experience", "wisdom", "insight",
        "深度思考", "專業判斷", "expert knowledge", "domain expertise"
    ],
    "Direct Communication": [
        "直接溝通", "direct communication", "straightforward", "transparent",
        "坦誠", "直率", "明確", "清楚", "不繞彎", "直截了當",
        "開誠布公", "honest communication", "clear and direct"
    ],
    "Open-Mindedness": [
        "開放心態", "open-minded", "開放性", "receptive", "flexible thinking",
        "包容", "接納", "多元思維", "diverse perspectives", "learning mindset",
        "成長心態", "growth mindset", "curiosity", "好奇心"
    ],
    "Ambition": [
        "企圖心", "ambition", "ambitious", "drive", "motivation",
        "積極進取", "上進心", "目標導向", "goal-oriented", "aspiration",
        "追求卓越", "excellence", "高標準", "high standards", "achievement"
    ]
}

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
    """Extracts text from PDF with enhanced encoding support."""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Check if text extraction was successful
                text += page_text + "\n"
        
        if not text.strip():
            return None  # 讓調用者處理錯誤顯示
        return text
    except Exception as e:
        st.error(f"PDF讀取錯誤: {e}")
        return None

def calculate_match_score(resume_text, jd_text):
    """Math-based ATS score."""
    text_list = [resume_text, jd_text]
    cv = TfidfVectorizer()
    count_matrix = cv.fit_transform(text_list)
    match_percentage = cosine_similarity(count_matrix)[0][1] * 100
    return round(match_percentage, 2)

def analyze_resume_general(resume_text, target_role=None):
    """Mode 1: General Analysis with optional target role."""
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
            
            # Prepare role-specific context and scoring criteria
            role_context = ""
            scoring_criteria = ""
            
            if target_role and target_role.strip():
                role_context = f"\n\nTARGET ROLE: {target_role.strip()}\nPlease provide recommendations specifically tailored for this role, including industry-specific keywords, relevant skills emphasis, and role-appropriate formatting suggestions."
                scoring_criteria = """
            **Overall Rating**: [Score out of 10 - Be STRICT: Consider format (3pts) + content relevance to target role (4pts) + ATS optimization (3pts)] ⭐
            **ATS Compatibility**: [High/Medium/Low - Consider keywords, format, sections for the target role] 🤖  
            **Role-Resume Match**: [High/Medium/Low/Mismatch - How well does the background align with target role?] 🎯"""
            else:
                scoring_criteria = """
            **Overall Rating**: [Score out of 10 - Be STRICT: Consider format (3pts) + content quality (4pts) + ATS optimization (3pts)] ⭐
            **ATS Compatibility**: [High/Medium/Low - Consider keywords, format, sections, readability] 🤖  
            **Professional Level**: [Entry/Mid/Senior - Based on experience depth and presentation quality] 📈"""
            
            prompt = f"""
            Act as a Senior Resume Writer and Career Strategist. Analyze the resume language and provide recommendations with cultural intelligence.
            
            LANGUAGE & CULTURAL INSTRUCTION: 
            - If the resume is in Chinese (Traditional or Simplified), respond ENTIRELY in Traditional Chinese
            - Use Chinese thinking patterns, workplace culture understanding, and local market insights
            - Consider Taiwan/Hong Kong/Chinese job market standards and expectations
            - Use appropriate Chinese business terminology and professional expressions
            - If in English, respond in English with Western business standards
            
            Resume Text:
            {resume_text[:2500]}
            
            Output in this EXACT format:
            
            ## 🎯 **Resume Score & Quick Assessment**
            
            {scoring_criteria}
            **Key Strength**: [One main strength in 5-7 words]
            **Priority Fix**: [Most urgent issue to address]
            
            ## 📝 **Top 3 Improvements (Priority Order)**
            
            ### 🥇 **#1 Critical Fix**
            **Issue**: [What's wrong]
            
            **Solution**: 
            [Specific action to take]
            
            ### 🥈 **#2 High Impact** 
            **Issue**: [What's wrong]
            
            **Solution**: 
            [Specific action to take]
            
            ### 🥉 **#3 Quick Win**
            **Issue**: [What's wrong]
            
            **Solution**: 
            [Specific action to take]
            
            ## 🚀 **Professional Summary Rewrite**
            
            **Current**: 
            [Brief assessment of existing summary]
            
            **Improved Version**: 
            [Write a compelling 2-sentence professional summary]
            
            ## 📊 **Content Enhancement**
            
            **Add These Keywords**: 
            [5-6 industry keywords]
            
            **Stronger Action Verbs**: 
            [Replace weak verbs with these 4-5 powerful alternatives]
            
            **Quantify These**: 
            [2-3 achievements that need numbers/metrics]
            
            ## ⚡ **30-Second Fixes**
            - [Quick formatting fix]
            - [Simple word replacement]
            - [Easy section adjustment]
            
            SCORING GUIDELINES:"""
            
            if target_role and target_role.strip():
                scoring_guidelines = """
            - If resume background completely mismatches target role: Overall Rating ≤ 4/10
            - If some transferable skills but different field: Overall Rating 4-6/10  
            - If relevant background with optimization needed: Overall Rating 6-8/10
            - If strong match with minor improvements: Overall Rating 8-10/10"""
            else:
                scoring_guidelines = """
            - Poor format, content quality, ATS issues: Overall Rating ≤ 4/10
            - Basic format, adequate content, some ATS optimization needed: Overall Rating 4-6/10  
            - Good format, solid content, minor ATS improvements: Overall Rating 6-8/10
            - Excellent format, strong content, ATS optimized: Overall Rating 8-10/10"""
            
            final_prompt = prompt + scoring_guidelines + role_context + """
            
            CULTURAL ADAPTATION FOR CHINESE RESUMES:
            When analyzing Chinese resumes, consider:
            - Traditional Chinese format preferences and professional presentation standards
            - Local market keyword preferences and industry terminology
            - Cultural nuances in self-presentation and achievement description
            - Regional job market expectations and hiring practices
            - Appropriate tone for Chinese professional communication
            """
            response = model.generate_content(final_prompt)
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
            Act as an Application Tracking System (ATS) and Technical Recruiter with cultural intelligence.
            Analyze both documents and respond with appropriate cultural and market context.
            
            LANGUAGE & CULTURAL INSTRUCTION:
            - If the resume is in Chinese (Traditional or Simplified), respond ENTIRELY in Traditional Chinese
            - Consider local job market standards, hiring practices, and cultural expectations
            - Use appropriate Chinese business terminology and recruitment language
            - If in English, respond in English with Western business standards
            
            Compare the Resume against the Job Description.
            
            Resume: {resume_text[:1800]}  # Reduced token usage
            Job Description: {jd_text[:800]}  # Reduced token usage
            
            Output in Markdown:
            1. **🚫 Missing Critical Skills**: List technical skills/tools in the JD that are NOT in the resume.
            2. **📉 Gap Analysis**: Briefly explain where the candidate's experience falls short of the requirements.
            3. **✍️ Bullet Point Rewrite**: Pick one existing bullet point from the resume and rewrite it to specifically target this job description using keywords from the JD.
            4. **⚖️ Final Verdict**: "High Match", "Medium Match", or "Low Match" with a 1-sentence reason.
            
            CULTURAL ADAPTATION FOR CHINESE CONTENT:
            When working with Chinese resumes, adapt recommendations to:
            - Local hiring manager expectations and evaluation criteria
            - Regional industry standards and keyword preferences  
            - Cultural communication styles and professional presentation norms
            - Market-specific skill emphasis and terminology
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

def analyze_appier_soft_skills(resume_text):
    """Mode 3: Appier Soft Skills Analysis - focused on 6 categories with 32 specific skills."""
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
            
            # Generate skill categories and lists for prompt
            skills_overview = ""
            for category, skills in APPIER_SOFT_SKILLS_FRAMEWORK.items():
                skills_overview += f"\n**{category}** ({len(skills)} skills):\n"
                for i, skill in enumerate(skills, 1):
                    skills_overview += f"   {i}. {skill}\n"
            
            prompt = f"""
            你是一位專業的人才評估專家，專門分析Appier公司的軟實力框架。請根據以下32項軟實力技能分析履歷內容。

            **Appier軟實力框架 (6大類別，32項技能):**
            {skills_overview}

            **評分標準:**
            - 使用1-10分制評分
            - 1-3分：無明顯證據或相關經驗
            - 4-6分：有基本相關經驗或潛力
            - 7-8分：有明確證據和具體經驗
            - 9-10分：有豐富經驗和卓越表現

            **履歷內容:**
            {resume_text[:3000]}

            **請按以下格式輸出分析結果:**

            ## 🎯 **Appier軟實力總評**

            **整體軟實力得分**: [總平均分數/10] 分
            **最強技能類別**: [得分最高的類別]
            **發展潛力**: [綜合評估：高/中/低]
            **Appier文化契合度**: [1-10分及簡要說明]

            ## 📊 **六大類別詳細分析**

            ### 🧠 **Problem Solving & Critical thinking** (平均分數: X.X/10)
            **識別到的技能:**
            - Problem Definition & Root Cause: [分數]/10 - [具體證據或"無明顯證據"]
            - Analytical & Logical Thinking: [分數]/10 - [具體證據或"無明顯證據"]  
            - Creative & Systems Thinking: [分數]/10 - [具體證據或"無明顯證據"]
            - Decision-Making & Execution: [分數]/10 - [具體證據或"無明顯證據"]
            - Adaptive & Collaborative Problem Solving: [分數]/10 - [具體證據或"無明顯證據"]

            **類別總評**: [整體表現描述]
            **發展建議**: [針對性改善建議]

            ### 💬 **Communication & Teamwork** (平均分數: X.X/10)
            **識別到的技能:**
            - Active Listening & Empathy: [分數]/10 - [具體證據或"無明顯證據"]
            - Clear & Impactful Expression: [分數]/10 - [具體證據或"無明顯證據"]
            - Persuasive & Stakeholder Communication: [分數]/10 - [具體證據或"無明顯證據"]
            - Feedback & Constructive Dialogue: [分數]/10 - [具體證據或"無明顯證據"]
            - Collaboration & Facilitation: [分數]/10 - [具體證據或"無明顯證據"]
            - Language Proficiency (Cross-Cultural Communication): [分數]/10 - [具體證據或"無明顯證據"]

            **類別總評**: [整體表現描述]
            **發展建議**: [針對性改善建議]

            ### 🤖 **AI Capability** (平均分數: X.X/10)
            **識別到的技能:**
            - AI Literacy & Awareness: [分數]/10 - [具體證據或"無明顯證據"]
            - Applied AI & Data-Driven Problem Solving: [分數]/10 - [具體證據或"無明顯證據"]
            - Ethical & Responsible AI Practice: [分數]/10 - [具體證據或"無明顯證據"]

            **類別總評**: [整體表現描述]
            **發展建議**: [針對性改善建議]

            ### 👑 **Leadership** (平均分數: X.X/10)
            **識別到的技能:**
            - Leading Self - Vision & Direction: [分數]/10 - [具體證據或"無明顯證據"]
            - Leading Self - Decision-Making Under Uncertainty: [分數]/10 - [具體證據或"無明顯證據"]
            - Leading Self - Personal Resilience & Sustainable Performance: [分數]/10 - [具體證據或"無明顯證據"]
            - Leading Others - People Management & Coaching: [分數]/10 - [具體證據或"無明顯證據"]
            - Leading Others - Stakeholder Influence & Executive Communication: [分數]/10 - [具體證據或"無明顯證據"]
            - Leading Others - Change Leadership: [分數]/10 - [具體證據或"無明顯證據"]
            - Leading Business - Strategic Thinking & Prioritization: [分數]/10 - [具體證據或"無明顯證據"]

            **類別總評**: [整體表現描述]
            **發展建議**: [針對性改善建議]

            ### 🌟 **Appier Leader** (平均分數: X.X/10)
            **識別到的技能:**
            - Growth & Development: [分數]/10 - [具體證據或"無明顯證據"]
            - Timely Feedback: [分數]/10 - [具體證據或"無明顯證據"]
            - Respect & Professionalism: [分數]/10 - [具體證據或"無明顯證據"]
            - Integrity & Trust: [分數]/10 - [具體證據或"無明顯證據"]
            - Genuine Care: [分數]/10 - [具體證據或"無明顯證據"]
            - Remove Obstacles: [分數]/10 - [具體證據或"無明顯證據"]
            - Focus & Priorities: [分數]/10 - [具體證據或"無明顯證據"]
            - Judgment & Expertise: [分數]/10 - [具體證據或"無明顯證據"]

            **類別總評**: [整體表現描述]
            **發展建議**: [針對性改善建議]

            ### 💎 **Appier Core Values** (平均分數: X.X/10)
            **識別到的技能:**
            - Direct Communication: [分數]/10 - [具體證據或"無明顯證據"]
            - Open-Mindedness: [分數]/10 - [具體證據或"無明顯證據"]
            - Ambition: [分數]/10 - [具體證據或"無明顯證據"]

            **類別總評**: [整體表現描述]
            **發展建議**: [針對性改善建議]

            ## 🚀 **個人發展建議**

            **立即可改善項目** (3個最需要的技能):
            1. [技能名稱] - [具體改善建議]
            2. [技能名稱] - [具體改善建議]  
            3. [技能名稱] - [具體改善建議]

            **中長期發展方向**:
            - [發展領域1]: [具體建議]
            - [發展領域2]: [具體建議]

            **Appier文化融入建議**:
            [針對Appier文化特色的具體建議]

            ## 📈 **團隊配置建議**

            **最適合的團隊角色**: [基於分析結果的建議]
            **可貢獻的價值**: [在團隊中能發揮的優勢]
            **需要的支持**: [團隊或組織應提供的協助]

            **注意**: 證據不足的技能並非代表不具備，可能是履歷呈現方式需要改善，或需要透過其他方式(面談、作品集等)進一步評估。
            """
            
            response = model.generate_content(prompt)
            increment_usage()  # Only count successful requests
            st.success(f"✅ Appier軟實力分析完成 (使用 {model_name})")
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
            Act as a Technical Talent Analyst and Skills Assessment Expert with cultural and market intelligence.
            Provide skills analysis with appropriate cultural context and market understanding.
            
            LANGUAGE & CULTURAL INSTRUCTION:
            - If the resume is in Chinese (Traditional or Simplified), respond ENTIRELY in Traditional Chinese
            - Consider local tech industry standards, skill naming conventions, and market demands
            - Use appropriate Chinese technical terminology and career development language
            - Understand regional differences in skill emphasis and career progression
            - If in English, respond in English with Western industry standards
            
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
            
            CULTURAL ADAPTATION FOR CHINESE CONTENT:
            When analyzing Chinese resumes, consider:
            - Local tech industry skill naming and categorization standards
            - Regional career progression patterns and expectations
            - Cultural context for skill presentation and team collaboration
            - Market-specific technology trends and demands
            - Appropriate Chinese technical and professional terminology
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
        ["📄 General Resume Review", "🎯 Compare with Job Description"]
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
    st.header("Resume Analysis & Enhancement")
    st.write("Upload your resume and choose the type of analysis that best fits your needs.")
    
    # Target role input
    target_role = st.text_input(
        "🎯 Target Role (Optional)", 
        placeholder="e.g., Software Engineer, Product Manager, Data Scientist...",
        help="Specify your target role for more personalized recommendations"
    )
    
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    
    if uploaded_file:
        # Display Appier framework overview for Appier analysis option
        with st.expander("📋 查看 Appier 軟實力框架 (6大類別, 32項技能)", expanded=False):
            for category, skills in APPIER_SOFT_SKILLS_FRAMEWORK.items():
                st.markdown(f"**{category}** ({len(skills)} 項技能):")
                for i, skill in enumerate(skills, 1):
                    st.markdown(f"   {i}. {skill}")
                st.markdown("")
        
        st.divider()
        st.subheader("Choose Analysis Type:")
        
        col1, col2 = st.columns(2)
        with col1:
            general_btn = st.button("📄 一般履歷優化", 
                                   type="primary",
                                   help="適合求職、轉職，專注ATS優化和履歷格式建議",
                                   use_container_width=True)
        with col2:
            appier_btn = st.button("🌟 Appier軟實力分析", 
                                  type="secondary",
                                  help="適合Appier員工，專注軟實力評估和職涯發展建議",
                                  use_container_width=True)
        
        # Initialize session state for analysis choice
        if 'analysis_choice' not in st.session_state:
            st.session_state.analysis_choice = None
        if 'analysis_result' not in st.session_state:
            st.session_state.analysis_result = None
            
        # Handle button clicks
        if general_btn:
            st.session_state.analysis_choice = "general"
            st.session_state.analysis_result = None
        elif appier_btn:
            st.session_state.analysis_choice = "appier"
            st.session_state.analysis_result = None
        
        # Process analysis based on choice
        if st.session_state.analysis_choice and st.session_state.analysis_result is None:
            with st.spinner("🔍 AI正在分析您的履歷..." if st.session_state.analysis_choice == "appier" else "🔍 AI is reviewing your profile..."):
                resume_text = extract_text_from_pdf(uploaded_file)
                
                # 如果PDF提取失敗，顯示替代方案
                if not resume_text:
                    st.error("🚨 **Cannot extract text from PDF!**" if st.session_state.analysis_choice == "general" else "🚨 **無法從PDF提取文字!**")
                    with st.expander("🛠️ **Alternative Solution - Click to expand**" if st.session_state.analysis_choice == "general" else "🛠️ **替代方案 - 點擊展開**", expanded=True):
                        if st.session_state.analysis_choice == "general":
                            st.markdown("""
                            **📄 This usually happens when:**
                            - 🖼️ **Scanned PDF**: Resume was scanned as an image
                            - 📷 **Image-based PDF**: Text is embedded in images, not as text
                            - 🔒 **Password-protected**: PDF has security restrictions
                            """)
                        else:
                            st.markdown("""
                            **📄 這種情況通常發生在:**
                            - 🖼️ **掃描的PDF**: 履歷是作為圖像掃描的
                            - 📷 **圖像化PDF**: 文字嵌入在圖像中，而不是文字
                            - 🔒 **密碼保護**: PDF有安全限制
                            """)
                        
                        manual_text_input = st.text_area(
                            "📝 **Paste your resume content here (Alternative)**" if st.session_state.analysis_choice == "general" else "📝 **在此貼上您的履歷內容**",
                            height=200,
                            placeholder="Copy and paste your resume content here..." if st.session_state.analysis_choice == "general" else "請複製並貼上您的履歷內容...",
                            help="For scanned PDFs or image-based resumes" if st.session_state.analysis_choice == "general" else "針對掃描或圖像化PDF的替代方案",
                            key="fallback_text_input"
                        )
                        
                        if manual_text_input.strip():
                            analyze_btn_text = "Analyze with Text Input" if st.session_state.analysis_choice == "general" else "用文字輸入分析"
                            if st.button(analyze_btn_text, type="primary"):
                                if st.session_state.analysis_choice == "general":
                                    analysis = analyze_resume_general(manual_text_input.strip(), target_role)
                                    if analysis:
                                        st.session_state.analysis_result = ("general", analysis)
                                        st.rerun()
                                else:  # appier
                                    analysis = analyze_appier_soft_skills(manual_text_input.strip())
                                    if analysis:
                                        st.session_state.analysis_result = ("appier", analysis)
                                        st.rerun()
                else:
                    # PDF extraction successful
                    if st.session_state.analysis_choice == "general":
                        analysis = analyze_resume_general(resume_text, target_role)
                        if analysis:
                            st.session_state.analysis_result = ("general", analysis)
                    else:  # appier
                        analysis = analyze_appier_soft_skills(resume_text)
                        if analysis:
                            st.session_state.analysis_result = ("appier", analysis)
        
        # Display results
        if st.session_state.analysis_result:
            analysis_type, analysis_content = st.session_state.analysis_result
            
            if analysis_type == "general":
                st.success("✅ **Analysis completed**")
                st.markdown("### 📝 Resume Optimization Report")
                st.markdown(analysis_content)
            else:  # appier
                st.success("✅ **軟實力分析完成**")
                st.markdown("### 🌟 Appier軟實力分析報告")
                st.markdown(analysis_content)
                
                # Export options for Appier analysis
                st.divider()
                st.subheader("📥 匯出選項")
                col1, col2 = st.columns(2)
                
                with col1:
                    text_content, filename = export_analysis_to_text(analysis_content, "appier_soft_skills_analysis")
                    st.download_button(
                        label="📄 下載完整報告 (TXT)",
                        data=text_content,
                        file_name=filename,
                        mime="text/plain"
                    )
                with col2:
                    st.info("🔄 JSON格式匯出\n(開發中)")
        
        # Reset button to choose different analysis
        if st.session_state.analysis_result:
            st.divider()
            if st.button("🔄 Choose Different Analysis", type="secondary"):
                st.session_state.analysis_choice = None
                st.session_state.analysis_result = None
                st.rerun()

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
                
                # 如果PDF提取失敗，顯示替代方案
                if not resume_text:
                    st.error("🚨 **Cannot extract text from PDF!**")
                    with st.expander("🛠️ **Alternative Solution - Click to expand**", expanded=True):
                        st.markdown("""
                        **📄 This usually happens when:**
                        - 🖼️ **Scanned PDF**: Resume was scanned as an image
                        - 📷 **Image-based PDF**: Text is embedded in images
                        - 🔒 **Password-protected**: PDF has security restrictions
                        """)
                        
                        manual_text_input = st.text_area(
                            "📝 **Paste your resume content here**",
                            height=150,
                            placeholder="Copy and paste your resume content here...",
                            key="fallback_text_input_mode2"
                        )
                        
                        if manual_text_input.strip() and st.button("Compare with Text Input", type="primary"):
                            with st.spinner("🤖 Analyzing with manual input..."):
                                # 1. Math Score
                                match_score = calculate_match_score(manual_text_input.strip(), jd_input)
                                
                                # 2. AI Analysis
                                ai_analysis = analyze_resume_vs_jd(manual_text_input.strip(), jd_input)
                                
                                # Display Results
                                st.divider()
                                st.subheader("📊 Analysis Results")
                                
                                # Math score
                                score_color = "red"
                                if match_score > 75: score_color = "green"
                                elif match_score > 50: score_color = "orange"
                                
                                st.markdown(f"### ATS Match Score: :{score_color}[{match_score}%]")
                                st.progress(int(match_score))
                                
                                # AI analysis
                                if ai_analysis:
                                    st.markdown("### 🤖 AI Gap Analysis")
                                    st.markdown(ai_analysis)
                                    st.success("✅ **Analysis completed using manual text input**")
                else:
                    # PDF extraction successful
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
            if not uploaded_file:
                st.error("⚠️ Please upload a PDF resume")
            if not jd_input:
                st.error("⚠️ Please enter a job description")