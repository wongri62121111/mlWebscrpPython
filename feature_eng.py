import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re

class FeatureEngineer:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100)
        self.svd = TruncatedSVD(n_components=10, random_state=42)
        self.skill_list = self._get_common_skills()
        
    def _get_common_skills(self):
        """Return a list of common skills in tech"""
        return [
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
        
    def create_skill_features(self, df):
        """Create binary features for skills mentioned in job descriptions"""
        print("Creating skill features...")
        
        # Initialize columns for each skill with 0s
        skill_df = pd.DataFrame(0, index=df.index, columns=self.skill_list)
        
        # For each job description, check for the presence of each skill
        for idx, row in df.iterrows():
            if pd.isna(row['description']):
                continue
                
            description = row['description'].lower()
            for skill in self.skill_list:
                # Check if skill is mentioned in the description
                if re.search(r'\b' + re.escape(skill.lower()) + r'\b', description):
                    skill_df.loc[idx, skill] = 1
                    
        # Merge the skill features with the original dataframe
        df_with_skills = pd.concat([df, skill_df], axis=1)
        print(f"Added {len(self.skill_list)} skill features")
        
        return df_with_skills
        
    def create_text_features(self, df, text_column='description'):
        """Create text features from job descriptions using TF-IDF and SVD"""
        print("Creating text features from job descriptions...")
        
        # Check if the column exists
        if text_column not in df.columns:
            print(f"Warning: {text_column} not found in dataframe")
            return df
        
        # Fill NA values
        descriptions = df[text_column].fillna('')
        
        # Apply TF-IDF
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(descriptions)
        
        # Apply dimensionality reduction with SVD
        svd_features = self.svd.fit_transform(tfidf_matrix)
        
        # Convert to dataframe
        svd_df = pd.DataFrame(
            svd_features, 
            columns=[f'desc_svd_{i}' for i in range(svd_features.shape[1])],
            index=df.index
        )
        
        # Merge with original dataframe
        df_with_text = pd.concat([df, svd_df], axis=1)
        print(f"Added {svd_features.shape[1]} text features")
        
        return df_with_text
        
    def create_location_features(self, df):
        """Create location-based features"""
        print("Creating location features...")
        
        # Check if location columns exist
        if 'city' not in df.columns or 'state' not in df.columns:
            print("Warning: location columns not found in dataframe")
            return df
        
        # Create city-state combination feature
        df['city_state'] = df['city'] + '_' + df['state']
        
        # Create indicator for major tech hubs
        tech_hubs = ['New York', 'San Francisco', 'Seattle', 'Boston', 'Austin', 'Los Angeles']
        df['is_tech_hub'] = df['city'].apply(lambda x: 1 if x in tech_hubs else 0)
        
        # Get dummies for state (if there are too many states, this might need to be limited)
        state_dummies = pd.get_dummies(df['state'], prefix='state')
        df = pd.concat([df, state_dummies], axis=1)
        print(f"Added location features including {len(state_dummies.columns)} state indicators")
        
        return df
        
    def create_company_features(self, df):
        """Create company-related features"""
        print("Creating company features...")
        
        # Check if company column exists
        if 'company' not in df.columns:
            print("Warning: company column not found in dataframe")
            return df
        
        # Create indicator for major tech companies
        tech_giants = ['Google', 'Amazon', 'Microsoft', 'Apple', 'Facebook', 'Meta', 'Netflix', 'Uber', 'Airbnb']
        df['is_tech_giant'] = df['company'].apply(lambda x: 1 if any(giant.lower() in str(x).lower() for giant in tech_giants) else 0)
        
        print("Added company features")
        return df
        
    def create_experience_features(self, df):
        """Create experience-related features"""
        print("Creating experience features...")
        
        # Check if experience column exists
        if 'experience_years' not in df.columns:
            print("Warning: experience_years column not found in dataframe")
            return df
        
        # Create experience level categories
        df['experience_level'] = pd.cut(
            df['experience_years'].fillna(0),
            bins=[-1, 0, 2, 5, 10, float('inf')],
            labels=['Not Specified', 'Entry Level', 'Mid Level', 'Senior', 'Expert']
        )
        
        # Get dummies for experience level
        exp_dummies = pd.get_dummies(df['experience_level'], prefix='exp')
        df = pd.concat([df, exp_dummies], axis=1)
        print(f"Added experience features including {len(exp_dummies.columns)} experience level indicators")
        
        return df
        
    def create_education_features(self, df):
        """Create education-related features"""
        print("Creating education features...")
        
        # Check if education column exists
        if 'education_level' not in df.columns:
            print("Warning: education_level column not found in dataframe")
            return df
        
        # Create education level ranking
        education_ranks = {
            'High School': 1,
            "Associate's": 2,
            "Bachelor's": 3,
            "Master's": 4,
            'PhD': 5,
            'Not Specified': 0
        }
        
        df['education_rank'] = df['education_level'].map(education_ranks).fillna(0)
        
        # Get dummies for education level
        edu_dummies = pd.get_dummies(df['education_level'], prefix='edu')
        df = pd.concat([df, edu_dummies], axis=1)
        print(f"Added education features including {len(edu_dummies.columns)} education level indicators")
        
        return df
        
    def create_salary_features(self, df):
        """Create salary-related features"""
        print("Creating salary features...")
        
        # Check if salary columns exist
        if 'salary_min' not in df.columns or 'salary_max' not in df.columns or 'salary_period' not in df.columns:
            print("Warning: salary columns not found in dataframe")
            return df
        
        # Convert hourly/monthly salaries to annual
        conversion_factors = {
            'hourly': 2080,  # 40 hours/week * 52 weeks
            'monthly': 12,   # 12 months/year
            'annual': 1,
            'unknown': 1
        }
        
        # Create average salary feature
        df['salary_avg'] = (df['salary_min'] + df['salary_max']) / 2
        
        # Convert to annual salary
        df['salary_annual'] = df.apply(
            lambda row: row['salary_avg'] * conversion_factors.get(row['salary_period'], 1),
            axis=1
        )
        
        # Create salary range feature
        df['salary_range'] = df['salary_max'] - df['salary_min']
        
        # Create salary range percentage
        df['salary_range_pct'] = df.apply(
            lambda row: (row['salary_range'] / row['salary_min']) * 100 if row['salary_min'] > 0 else 0,
            axis=1
        )
        
        print("Added salary features")
        return df
        
    def engineer_features(self, df):
        """Apply all feature engineering transformations"""
        # Create skill features
        df = self.create_skill_features(df)
        
        # Create text features from job descriptions
        df = self.create_text_features(df)
        
        # Create location features
        df = self.create_location_features(df)
        
        # Create company features
        df = self.create_company_features(df)
        
        # Create experience features
        df = self.create_experience_features(df)
        
        # Create education features
        df = self.create_education_features(df)
        
        # Create salary features
        df = self.create_salary_features(df)
        
        print(f"Feature engineering complete. Final dataframe shape: {df.shape}")
        return df