import pandas as pd
import random
from faker import Faker

# Initialize Faker for generating fake data
fake = Faker()

# Define lists of possible values
job_titles = [
    "Software Engineer", "Data Scientist", "Machine Learning Engineer",
    "DevOps Engineer", "Frontend Developer", "Backend Developer",
    "Full Stack Developer", "Product Manager", "UX/UI Designer",
    "Cloud Engineer", "Database Administrator", "Cybersecurity Analyst"
]

companies = [
    "TechCorp", "DataWorks", "Cloudify", "CodeMasters", "InnovateX",
    "FutureTech", "AI Solutions", "WebWizards", "SecureIT", "DevOpsPro"
]

locations = [
    "New York, NY", "San Francisco, CA", "Los Angeles, CA", "Chicago, IL",
    "Austin, TX", "Seattle, WA", "Boston, MA", "Denver, CO", "Atlanta, GA",
    "Miami, FL"
]

skills = [
    "Python", "Java", "JavaScript", "C++", "C#", "SQL", "NoSQL", "MongoDB",
    "React", "Angular", "Vue", "Node.js", "AWS", "Azure", "GCP",
    "Docker", "Kubernetes", "Git", "CI/CD", "Jenkins", "TensorFlow",
    "PyTorch", "Scikit-learn", "Pandas", "NumPy", "R", "Tableau",
    "Power BI", "Excel", "Machine Learning", "Deep Learning", "NLP",
    "Computer Vision", "Data Analysis", "Data Visualization", "Statistics",
    "Agile", "Scrum", "Project Management", "REST API", "GraphQL",
    "Django", "Flask", "Spring", "Hibernate", "TypeScript", "Go", "Rust",
    "Swift", "Kotlin", "PHP", "Ruby", "Rails", "Linux", "Unix", "Windows Server"
]

sources = ["CareerBuilder", "SimplyHired", "Glassdoor"]

# Function to generate a single job posting
def generate_job_posting():
    job_title = random.choice(job_titles)
    company = random.choice(companies)
    location = random.choice(locations)
    salary_min = random.randint(50000, 100000)  # Minimum salary
    salary_max = salary_min + random.randint(10000, 30000)  # Maximum salary
    salary = f"${salary_min} - ${salary_max}"  # Simulate a salary range
    description = fake.paragraph(nb_sentences=5)  # Generate a fake job description
    num_skills = random.randint(3, 10)  # Random number of skills
    job_skills = random.sample(skills, num_skills)  # Randomly select skills
    source = random.choice(sources)  # Randomly select a source

    return {
        "title": job_title,
        "company": company,
        "location": location,
        "salary": salary,  # Use the salary range
        "description": description,
        "skills": ", ".join(job_skills),  # Convert list of skills to a comma-separated string
        "source": source,
    }
# Generate 4000 job postings
data = [generate_job_posting() for _ in range(4000)]

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("job_data.csv", index=False)
print("Generated 4000 synthetic job postings and saved to 'job_data.csv'.")