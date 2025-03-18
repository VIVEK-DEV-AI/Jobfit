import os
import pandas as pd
import pickle
from pypdf import PdfReader
import re
import streamlit as st
import spacy

# Load models
word_vector = pickle.load(open("tfidf1.pkl", "rb"))
model = pickle.load(open("model1.pkl", "rb"))

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def extract_candidate_info(text):
    name = extract_name(text)
    contact = extract_contact_details(text)
    skills = extract_skills(text)
    return name, contact, skills

def extract_name(text):
    # Extract the first line of the resume as the name
    first_line = text.split('\n')[0].strip()
    
    # If "Name:" is present, remove it
    if first_line.lower().startswith("name:"):
        first_line = first_line[5:].strip()  # Remove the "Name:" part
    
    # Split by comma or new line and take the first part
    name = re.split(r'[,\n]', first_line)[0].strip()
    
    return name

def extract_contact_details(text):
    phone = re.findall(r'\b\d{10}\b', text)
    email = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    return {'phone': phone[0] if phone else '', 'email': email[0] if email else ''}

def extract_skills(text):
    # List of common skill-related terms to catch, can be expanded or customized
    skill_patterns = [
        'Java', 'Python', 'C++', 'SQL', 'HTML', 'CSS', 'JavaScript', 'Excel', 'Tableau', 'Pandas', 'Numpy', 'Matplotlib', 
        'React', 'Node.js', 'AWS', 'Azure', 'Docker', 'Kubernetes', 'Git', 'Jenkins', 'TensorFlow', 'PyTorch', 'Scala'
    ]
    
    skills_found = []

    # Use spaCy to extract entities and patterns that could be skills
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == 'ORG' or ent.label_ == 'PRODUCT':  # We assume skills can sometimes be products/organizations
            skills_found.append(ent.text)
    
    # Also check for predefined skills based on patterns
    for skill in skill_patterns:
        if skill.lower() in text.lower():
            skills_found.append(skill)
    
    # Clean skills: Remove non-alphanumeric characters and duplicates
    skills_found_cleaned = [skill.strip() for skill in skills_found]
    skills_found_cleaned = list(set([re.sub(r'[^a-zA-Z0-9\s]', '', skill) for skill in skills_found_cleaned]))
    
    # Remove any empty strings
    skills_found_cleaned = [skill for skill in skills_found_cleaned if skill]

    return ', '.join(skills_found_cleaned)  # Return cleaned skills

category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

def categorize_resumes(uploaded_files, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    results = []
    
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.pdf'):
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            cleaned_resume = cleanResume(text)

            input_features = word_vector.transform([cleaned_resume])
            prediction_id = model.predict(input_features)[0]
            category_name = category_mapping.get(prediction_id, "Unknown")
            
            category_folder = os.path.join(output_directory, category_name)
            
            if not os.path.exists(category_folder):
                os.makedirs(category_folder)
            
            target_path = os.path.join(category_folder, uploaded_file.name)
            with open(target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            name, contact, skills = extract_candidate_info(text)
            results.append({
                'name': name,
                'category': category_name,
                'filename': uploaded_file.name,
                'skills': skills,
                'contact': f"{contact['phone']} / {contact['email']}",
                'resume': cleaned_resume
            })
    
    results_df = pd.DataFrame(results)
    return results_df

def filter_resumes_by_skills(resumes_df, skills):
    if skills:
        skills = skills.lower().split(',')
        filtered_df = resumes_df[resumes_df['resume'].apply(lambda x: all(skill.strip() in x.lower() for skill in skills))]
        return filtered_df
    return resumes_df

# Inject custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #121212;
    }
    .stApp {
        max-width: 900px;
        margin: auto;
        padding: 30px;
        background-color: #1e1e1e;
        border-radius: 12px;
        box-shadow: 0px 0px 15px rgba(0, 255, 255, 0.1);
    }
    .stButton>button {
        background-color: #009688;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 12px 25px;
        font-size: 18px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00796b;
    }
    .stTextInput>div>input {
        border: 2px solid #009688;
        border-radius: 6px;
        padding: 12px;
        font-size: 18px;
        width: 100%;
        background-color: #121212;
        color: white;
        margin-bottom: 20px;
    }
    .stFileUploader>label {
        font-size: 18px;
        color: #009688;
        margin-bottom: 10px;
    }
    h1, h2, h3 {
        text-align: center;
        color: #00bcd4;
    }
    .stTextInput, .stFileUploader {
        margin-bottom: 20px;
    }
    .stButton {
        margin-top: 20px;
    }
    .stMarkdown {
        font-family: "Roboto", sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>**Resume Categorizer Application**</h1>", unsafe_allow_html=True)
st.markdown("<h3>With Python & Machine Learning</h3>", unsafe_allow_html=True)
st.markdown("This application categorizes resumes and extracts key information like name, contact details, and skills.")

uploaded_files = st.file_uploader("**Choose PDF files**", type="pdf", accept_multiple_files=True)
output_directory = st.text_input("**Output Directory**", "categorized_resumes")
skills_input = st.text_input("**Enter Skills (comma-separated)**")

if st.button("Categorize Resumes"):
    if uploaded_files and output_directory:
        results_df = categorize_resumes(uploaded_files, output_directory)
        
        filtered_df = filter_resumes_by_skills(results_df, skills_input)
        
        st.write(filtered_df[['filename', 'category']])
        
        results_csv = filtered_df[['name', 'category', 'filename', 'skills', 'contact']].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download results as CSV",
            data=results_csv,
            file_name='filtered_categorized_resumes.csv',
            mime='text/csv',
        )
        st.success("Resumes categorization and processing completed.")
    else:
        st.error("Please upload files and specify the output directory.")
