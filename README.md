# AI Resume Analyzer

A powerful AI-powered resume analysis tool built with Streamlit and Google Gemini AI.

## Features

- 📄 **General Resume Review**: Get insights on your strengths, weaknesses, and suitable roles
- 🎯 **Job Fit Analysis**: Compare your resume against specific job descriptions
- 📊 **ATS Score Calculation**: Mathematical matching score using TF-IDF and cosine similarity
- 🤖 **AI Analysis**: Detailed feedback powered by Google Gemini AI

## Live Demo

🌐 **[Try the app here](https://your-app-url.streamlit.app)**

## How to Use

1. **General Resume Review**:
   - Upload your PDF resume
   - Get professional analysis and recommendations

2. **Job Fit Analysis**:
   - Upload your resume + paste job description
   - Get matching score and targeted improvement suggestions

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Configuration

Set your Google Gemini API key in Streamlit secrets:
```toml
[secrets]
GOOGLE_API_KEY = "your-api-key-here"
```