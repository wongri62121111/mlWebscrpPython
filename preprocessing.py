import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.standard_scaler = StandardScaler()
        
    def load_data(self, file_path):
        """Load data from CSV file"""
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide CSV or JSON file")
        
        print(f"Loaded {len(df)} rows from {file_path}")
        return df
        
    def clean_data(self, df):
        """Clean the loaded data"""
        print("Cleaning data...")
        
        # Drop duplicates
        df_clean = df.drop_duplicates()
        print(f"Removed {len(df) - len(df_clean)} duplicate rows")
        
        # Handle missing values
        df_clean['title'] = df_clean['title'].fillna('Unknown')
        df_clean['company'] = df_clean['company'].fillna('Unknown')
        df_clean['location'] = df_clean['location'].fillna('Unknown')
        df_clean['description'] = df_clean['description'].fillna('')
        
        # Extract salary information and convert to numeric
        df_clean['salary_min'], df_clean['salary_max'], df_clean['salary_period'] = zip(*df_clean['salary'].apply(self._parse_salary))
        
        # Standardize job titles
        df_clean['standardized_title'] = df_clean['title'].apply(self._standardize_job_title)
        
        # Extract years of experience from description
        df_clean['experience_years'] = df_clean['description'].apply(self._extract_experience)
        
        # Extract education level from description
        df_clean['education_level'] = df_clean['description'].apply(self._extract_education)
        
        # Standardize location
        df_clean['city'], df_clean['state'] = zip(*df_clean['location'].apply(self._standardize_location))
        
        return df_clean
        
    def _parse_salary(self, salary_text):
        """Parse salary text into min, max, and period."""
        if pd.isna(salary_text) or not salary_text:
            return np.nan, np.nan, 'unknown'

        # If salary is already a number, return it directly
        if isinstance(salary_text, (int, float)):
            return salary_text, salary_text, 'annual'

        # Convert to lowercase for easier pattern matching
        salary_text = str(salary_text).lower()

        # Try to find salary ranges (e.g., "$50,000 - $70,000")
        range_pattern = r'(\$[\d,]+\.?\d*)\s*-\s*(\$[\d,]+\.?\d*)'
        range_match = re.search(range_pattern, salary_text)

        # Try to find single salary figures (e.g., "$60,000")
        single_pattern = r'(\$[\d,]+\.?\d*)'
        single_match = re.search(single_pattern, salary_text)

        # Determine period (annual, monthly, hourly)
        period = 'annual'  # Default
        if 'hour' in salary_text or 'hourly' in salary_text:
            period = 'hourly'
        elif 'month' in salary_text or 'monthly' in salary_text:
            period = 'monthly'
        elif 'year' in salary_text or 'annual' in salary_text or 'annually' in salary_text:
            period = 'annual'

        # Extract salary values
        if range_match:
            min_salary = float(range_match.group(1).replace('$', '').replace(',', ''))
            max_salary = float(range_match.group(2).replace('$', '').replace(',', ''))
            return min_salary, max_salary, period
        elif single_match:
            salary = float(single_match.group(1).replace('$', '').replace(',', ''))
            return salary, salary, period
        else:
            return np.nan, np.nan, 'unknown'
            
    def _standardize_job_title(self, title):
        """Standardize job titles to common categories"""
        if pd.isna(title) or not title:
            return 'Unknown'
            
        title = title.lower()
        
        # Define standardization rules for common job titles
        if any(term in title for term in ['software engineer', 'developer', 'programmer', 'coder', 'swe']):
            return 'Software Engineer'
        elif any(term in title for term in ['data scientist', 'ds', 'data science']):
            return 'Data Scientist'
        elif any(term in title for term in ['data analyst', 'business analyst', 'analytics']):
            return 'Data Analyst'
        elif any(term in title for term in ['machine learning', 'ml engineer', 'ai engineer']):
            return 'Machine Learning Engineer'
        elif any(term in title for term in ['product manager', 'pm', 'product owner']):
            return 'Product Manager'
        elif any(term in title for term in ['ux', 'ui designer', 'user interface', 'user experience']):
            return 'UX/UI Designer'
        elif any(term in title for term in ['devops', 'dev ops', 'sre', 'reliability', 'cloud engineer']):
            return 'DevOps Engineer'
        elif any(term in title for term in ['frontend', 'front end', 'front-end']):
            return 'Frontend Engineer'
        elif any(term in title for term in ['backend', 'back end', 'back-end']):
            return 'Backend Engineer'
        elif any(term in title for term in ['fullstack', 'full stack', 'full-stack']):
            return 'Fullstack Engineer'
        else:
            return 'Other'
            
    def _extract_experience(self, description):
        """Extract years of experience from job description"""
        if pd.isna(description) or not description:
            return np.nan
            
        description = description.lower()
        
        # Look for patterns like "X years of experience" or "X+ years"
        patterns = [
            r'(\d+)(?:\+)?\s*(?:-\s*\d+)?\s*years?(?:\s*of)?\s*experience',
            r'experience(?:\s*of)?\s*(\d+)(?:\+)?\s*(?:-\s*\d+)?\s*years',
            r'(\d+)(?:\+)?\s*years?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, description)
            if match:
                try:
                    return float(match.group(1))
                except (IndexError, ValueError):
                    continue
                    
        return np.nan
        
    def _extract_education(self, description):
        """Extract education level from job description"""
        if pd.isna(description) or not description:
            return 'Not Specified'
            
        description = description.lower()
        
        if any(term in description for term in ['phd', 'ph.d', 'doctorate', 'doctoral']):
            return 'PhD'
        elif any(term in description for term in ['master', 'ms', 'm.s', 'msc', 'm.sc']):
            return "Master's"
        elif any(term in description for term in ['bachelor', 'bs', 'b.s', 'ba', 'b.a', 'undergraduate']):
            return "Bachelor's"
        elif any(term in description for term in ['associate', 'aa', 'a.a', 'a.s']):
            return "Associate's"
        elif any(term in description for term in ['high school', 'highschool', 'ged']):
            return 'High School'
        else:
            return 'Not Specified'
            
    def _standardize_location(self, location):
        """Standardize location into city and state"""
        if pd.isna(location) or not location:
            return 'Unknown', 'Unknown'
            
        # Check if location contains a comma (City, State format)
        if ',' in location:
            parts = [part.strip() for part in location.split(',')]
            city = parts[0]
            
            # Try to extract state from the second part
            state_pattern = r'([A-Z]{2})'
            match = re.search(state_pattern, parts[1])
            if match:
                state = match.group(1)
            else:
                state = parts[1]
        else:
            # If no comma, assume it's just a city or just a state
            city = location
            state = 'Unknown'
            
        return city, state
        
    def encode_categorical_features(self, df, columns):
        """Encode categorical features using Label Encoding"""
        df_encoded = df.copy()
        
        for col in columns:
            if col in df.columns:
                le = LabelEncoder()
                df_encoded[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                
        return df_encoded
        
    def scale_numerical_features(self, df, columns):
        """Scale numerical features using StandardScaler"""
        df_scaled = df.copy()
        
        # Filter columns that exist in the dataframe
        valid_columns = [col for col in columns if col in df.columns]
        
        if valid_columns:
            df_valid = df[valid_columns].copy()
            # Fill NaN values with mean for scaling
            df_valid = df_valid.fillna(df_valid.mean())
            scaled_features = self.standard_scaler.fit_transform(df_valid)
            
            for i, col in enumerate(valid_columns):
                df_scaled[f'{col}_scaled'] = scaled_features[:, i]
                
        return df_scaled
        
    def clean_text(self, text):
        """Clean and tokenize text data"""
        if pd.isna(text) or not text:
            return []
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        return tokens

    def process_data(self, file_path):
        """Complete preprocessing pipeline"""
        # Load data
        df = self.load_data(file_path)
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Encode categorical features
        categorical_columns = ['standardized_title', 'company', 'city', 'state', 'education_level', 'salary_period', 'source']
        df_encoded = self.encode_categorical_features(df_clean, categorical_columns)
        
        # Scale numerical features
        numerical_columns = ['salary_min', 'salary_max', 'experience_years']
        df_scaled = self.scale_numerical_features(df_encoded, numerical_columns)
        
        print(f"Preprocessing complete. Final dataframe shape: {df_scaled.shape}")
        return df_scaled