import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance
import os

class DataVisualizer:
    def __init__(self, output_dir='visualizations'):
        """Initialize the data visualizer"""
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set default style
        sns.set(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
    def plot_salary_distribution(self, df, column='salary_annual', by_category=None):
        """Plot histogram of salary distribution"""
        print(f"Plotting salary distribution by {by_category if by_category else 'overall'}...")
        
        plt.figure(figsize=(12, 8))
    
        if by_category and by_category in df.columns:
            # Filter out rows with NaN values in the target column
            df_filtered = df[df[column].notna()]
            
            # Get the top categories by frequency
            top_categories = df_filtered[by_category].value_counts().nlargest(5).index
            
            # Filter the DataFrame to only include the top categories
            df_plot = df_filtered[df_filtered[by_category].isin(top_categories)]
            
            # Create a histogram for each category
            for category in top_categories:
                subset = df_plot[df_plot[by_category] == category]
                sns.histplot(subset[column], kde=True, label=category)
                
            plt.legend()
            plt.title(f'Salary Distribution by {by_category}')
        else:
            # Overall distribution
            sns.histplot(df[column].dropna(), kde=True)
            plt.title('Overall Salary Distribution')
            
        plt.xlabel('Annual Salary ($)')
        plt.ylabel('Frequency')
        
        # Save the figure
        filename = f'salary_dist_by_{by_category}.png' if by_category else 'salary_dist_overall.png'
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        
    def plot_salary_boxplot(self, df, column='salary_annual', by_category=None):
        """Plot boxplot of salary distribution by category"""
        print(f"Plotting salary boxplot by {by_category}...")
        
        if not by_category or by_category not in df.columns:
            print(f"Category {by_category} not found in dataframe")
            return
            
        plt.figure(figsize=(14, 8))
        
        # Filter out rows with NaN values
        df_filtered = df[df[column].notna()]
        
        # Get the top categories by frequency
        top_categories = df_filtered[by_category].value_counts().nlargest(8).index
        
        # Filter the DataFrame to only include the top categories
        df_plot = df_filtered[df_filtered[by_category].isin(top_categories)]
        
        # Create the boxplot
        sns.boxplot(x=by_category, y=column, data=df_plot)
        
        plt.title(f'Salary Distribution by {by_category}')
        plt.xlabel(by_category)
        plt.ylabel('Annual Salary ($)')
        plt.xticks(rotation=45)
        
        # Save the figure
        plt.tight_layout()
        filename = f'salary_boxplot_by_{by_category}.png'
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        
    def plot_salary_map(self, df, location_column='state', salary_column='salary_annual'):
        """Plot average salary by location on a map (US only)"""
        print("Plotting salary map...")
        
        # Check if necessary columns exist
        if location_column not in df.columns or salary_column not in df.columns:
            print(f"Required columns not found in dataframe")
            return
            
        # Calculate average salary by location
        location_salary = df.groupby(location_column)[salary_column].mean().reset_index()
        
        # Create the map
        fig = px.choropleth(
            location_salary,
            locations=location_column,
            color=salary_column,
            locationmode='USA-states',
            scope="usa",
            color_continuous_scale="Viridis",
            title=f'Average Salary by {location_column}',
            labels={salary_column: 'Average Annual Salary ($)'}
        )
        
        # Save the figure
        filename = f'salary_map_by_{location_column}.html'
        fig.write_html(os.path.join(self.output_dir, filename))
        
        print(f"Map saved to {filename}")
        
    def plot_skill_importance(self, model, feature_names, title='Skill Importance'):
        """Plot feature importance for skills"""
        print("Plotting skill importance...")
        
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            print("Model doesn't have feature importances attribute")
            return
            
        # Create DataFrame with feature names and importances
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(12, 10))
        sns.barplot(x='importance', y='feature', data=importance_df.head(20))
        
        plt.title(title)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        
        # Save the figure
        filename = f'{title.lower().replace(" ", "_")}.png'
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        
        return importance_df
        
    def plot_skill_heatmap(self, df, job_roles, skills):
        """Plot heatmap of skill importance across different job roles"""
        print("Plotting skill heatmap...")
        
        # Initialize a DataFrame to store the counts
        heatmap_data = pd.DataFrame(index=job_roles, columns=skills, data=0)
        
        # For each job role, count the occurrence of each skill
        for role in job_roles:
            role_df = df[df['standardized_title'] == role]
            total_jobs = len(role_df)
            
            if total_jobs == 0:
                continue
                
            for skill in skills:
                if skill in df.columns:
                    skill_count = role_df[skill].sum()
                    heatmap_data.loc[role, skill] = (skill_count / total_jobs) * 100
                    
        # Plot the heatmap
        plt.figure(figsize=(16, 10))
        sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.1f')
        
        plt.title('Skill Importance Across Job Roles (%)')
        plt.ylabel('Job Role')
        plt.xlabel('Skill')
        
        # Save the figure
        plt.tight_layout()
        filename = 'skill_heatmap.png'
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        
        return heatmap_data
        
    def plot_predicted_vs_actual(self, y_true, y_pred, title='Predicted vs Actual'):
        """Plot predicted vs actual values"""
        print("Plotting predicted vs actual values...")
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Plot the perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title(title)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        
        # Save the figure
        filename = f'{title.lower().replace(" ", "_")}.png'
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        
    def plot_residuals(self, y_true, y_pred, title='Residuals'):
        """Plot residuals"""
        print("Plotting residuals...")
        
        residuals = y_true - y_pred
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        
        # Save the figure
        filename = f'{title.lower().replace(" ", "_")}.png'
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        
    def visualize_data(self, df, model1=None, model2=None, X=None, y=None, feature_names=None):
        """Create a comprehensive set of visualizations"""
        # Salary distribution plots
        self.plot_salary_distribution(df)
        self.plot_salary_distribution(df, by_category='standardized_title')
        
        # Salary boxplots
        self.plot_salary_boxplot(df, by_category='standardized_title')
        self.plot_salary_boxplot(df, by_category='experience_level')
        self.plot_salary_boxplot(df, by_category='education_level')
        
        # Salary map
        self.plot_salary_map(df)
        
        # If models and feature names are provided, plot importance
        if model1 is not None and feature_names is not None:
            self.plot_skill_importance(model1, feature_names, 'Model 1 Feature Importance')
        
        if model2 is not None and feature_names is not None:
            self.plot_skill_importance(model2, feature_names, 'Model 2 Feature Importance')
        
        # Skill heatmap for top job roles
        top_roles = df['standardized_title'].value_counts().nlargest(5).index.tolist()
        skill_cols = [col for col in df.columns if col in self._get_common_skills()]
        
        if skill_cols and len(top_roles) > 0:
            self.plot_skill_heatmap(df, top_roles, skill_cols[:20])  # Use top 20 skills
        
        # If actual and predicted values are provided, plot those
        if X is not None and y is not None and model1 is not None:
            y_pred1 = model1.predict(X)
            self.plot_predicted_vs_actual(y, y_pred1, 'Model 1: Predicted vs Actual')
            self.plot_residuals(y, y_pred1, 'Model 1: Residuals')
        
        if X is not None and y is not None and model2 is not None:
            y_pred2 = model2.predict(X)
            self.plot_predicted_vs_actual(y, y_pred2, 'Model 2: Predicted vs Actual')
            self.plot_residuals(y, y_pred2, 'Model 2: Residuals')
        
        print("All visualizations completed")
        
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