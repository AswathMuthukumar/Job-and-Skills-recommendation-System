import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Sample dataset with job titles, skills, and associated companies
data = {
    'job_title': [
        'Data Scientist', 'Software Engineer', 'Product Manager',
        'Graphic Designer', 'Machine Learning Engineer', 'Data Analyst',
        'DevOps Engineer', 'Business Analyst', 'UX/UI Designer',
        'Cybersecurity Specialist', 'Web Developer', 'Cloud Architect',
        'Database Administrator', 'Mobile App Developer', 'SEO Specialist',
        'Network Engineer', 'IT Support Specialist', 'Technical Writer',
        'AI Researcher', 'Blockchain Developer', 'Game Developer',
        'Digital Marketing Manager', 'Embedded Systems Engineer', 
        'Electrical Engineer', 'Civil Engineer', 'Mechanical Engineer'
    ],
    'skills': [
        'Python, Machine Learning, Data Analysis, SQL, Statistics',
        'Java, Software Development, SQL, Python, Problem-Solving',
        'Project Management, Communication, Agile, Roadmapping, Strategy',
        'Adobe Photoshop, Creativity, Graphic Design, Illustrator, UX',
        'Python, TensorFlow, Data Science, Deep Learning, AI',
        'Data Analysis, Excel, Python, SQL, Visualization',
        'Linux, Jenkins, Kubernetes, Cloud, CI/CD, Scripting',
        'Business Analysis, Data Analysis, Communication, Presentation',
        'UX Research, Wireframing, Prototyping, Figma, Adobe XD',
        'Network Security, Risk Assessment, Firewalls, Ethical Hacking',
        'HTML, CSS, JavaScript, Frontend Development, Responsive Design',
        'Cloud Computing, AWS, Azure, Network Architecture, Automation',
        'SQL, Database Design, Performance Tuning, Backup, Recovery',
        'Java, Kotlin, Android Development, iOS Development, APIs',
        'SEO, Google Analytics, SEM, Content Marketing, Keyword Research',
        'Networking, Cisco, Troubleshooting, Routing, Firewalls',
        'Troubleshooting, Technical Support, Networking, Hardware',
        'Technical Writing, Documentation, Communication, Research',
        'Artificial Intelligence, Machine Learning, Neural Networks, NLP',
        'Blockchain, Solidity, Cryptography, Smart Contracts, Ethereum',
        'C++, Unity, Game Design, 3D Modeling, Unreal Engine',
        'Digital Marketing, SEO, Google Ads, Content Strategy, Social Media',
        'C, C++, Microcontrollers, Embedded Systems, RTOS, ARM',
        'Circuit Design, Electrical Engineering, MATLAB, PCB Layout',
        'Civil Engineering, AutoCAD, Project Management, Structural Analysis',
        'Mechanical Engineering, CAD, Product Design, FEA, Thermodynamics'
    ],
    'company': [
        'Google', 'Microsoft', 'Amazon', 'Apple', 'Facebook', 'IBM', 'Netflix', 
        'Deloitte', 'Spotify', 'Cisco', 'Shopify', 'Oracle', 'Salesforce', 
        'Uber', 'LinkedIn', 'Intel', 'Tesla', 'Slack', 'OpenAI', 'Coinbase', 
        'Riot Games', 'HubSpot', 'Qualcomm', 'GE', 'Fluor Corporation', 'Boeing'
    ]
}

# Creating the DataFrame
jobs_df = pd.DataFrame(data)

# Function to recommend jobs based on user input
def recommend_jobs_and_skills(user_input_skills, jobs_df):
    # Normalize user input
    user_input_skills = user_input_skills.lower().strip()

    # Combine job title and skills into a single column for processing
    jobs_df['combined'] = jobs_df['job_title'].str.lower() + ' ' + jobs_df['skills'].str.lower()
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(jobs_df['combined'])

    # Transform the user input for comparison
    user_tfidf = tfidf.transform([user_input_skills])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    
    # Get the indices of the top 5 recommendations
    recommended_indices = cosine_sim.argsort()[-5:][::-1]
    
    # Get the recommended job titles, skills, and companies
    recommendations = jobs_df.iloc[recommended_indices]
    
    return recommendations[['job_title', 'skills', 'company']].to_dict(orient='records')

# Function to get skills for a specific job title
def get_skills_for_job(job_title, jobs_df):
    job = jobs_df[jobs_df['job_title'].str.lower() == job_title.lower()]
    if not job.empty:
        return job['skills'].values[0]
    else:
        return "Job title not found."

# Function to get company for a specific job title
def get_company_for_job(job_title, jobs_df):
    job = jobs_df[jobs_df['job_title'].str.lower() == job_title.lower()]
    if not job.empty:
        return job['company'].values[0]
    else:
        return "Job title not found."

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = None
    skills_needed = None
    company_name = None

    if request.method == 'POST':
        choice = request.form['choice']
        
        if choice == 'recommendations':
            user_skills = request.form['skills']
            if user_skills:
                recommendations = recommend_jobs_and_skills(user_skills, jobs_df)
        
        elif choice == 'skills':
            job_title_input = request.form['job_title']
            skills_needed = get_skills_for_job(job_title_input, jobs_df)
        
        elif choice == 'company':
            job_title_input = request.form['job_title']
            company_name = get_company_for_job(job_title_input, jobs_df)

    return render_template('index.html', recommendations=recommendations, skills_needed=skills_needed, company_name=company_name)

if __name__ == '__main__':
    app.run(debug=True)
