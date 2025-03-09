import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

class ModelTrainer:
    def __init__(self, output_dir='models'):
        """Initialize the model trainer"""
        self.output_dir = output_dir
        self.models = {}
        self.feature_importance = {}
        self.metrics = {}

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def prepare_data(self, df, target_column='salary_annual', test_size=0.2, random_state=42):
        """Prepare data for model training"""
        print("Preparing data for model training...")

        # Check if target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")

        # Remove rows with NaN in target column
        df_clean = df.dropna(subset=[target_column])
        print(f"Removed {len(df) - len(df_clean)} rows with NaN in target column")

        # Identify numerical and categorical columns
        numerical_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()

        # Remove target column from features
        if target_column in numerical_cols:
            numerical_cols.remove(target_column)

        # Prepare features and target
        X = df_clean[numerical_cols]
        y = df_clean[target_column]

        # Drop columns with all NaN values
        X = X.dropna(axis=1, how='all')
        print(f"Dropped {len(numerical_cols) - len(X.columns)} columns with all NaN values")

        # Handle remaining missing values in numerical features
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"Data prepared. Training set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test, X.columns.tolist()

    def train_linear_model(self, X_train, y_train, model_name='linear_regression'):
        """Train a linear regression model"""
        print(f"Training {model_name} model...")

        # Initialize and train the model
        if model_name == 'linear_regression':
            model = LinearRegression()
        elif model_name == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_name == 'lasso':
            model = Lasso(alpha=0.1)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        model.fit(X_train, y_train)

        # Save the trained model
        self.models[model_name] = model

        # Save model to disk
        joblib.dump(model, os.path.join(self.output_dir, f"{model_name}.pkl"))

        print(f"{model_name} model trained and saved")
        return model

    def train_tree_model(self, X_train, y_train, model_name='random_forest'):
        """Train a tree-based model"""
        print(f"Training {model_name} model...")

        # Initialize and train the model
        if model_name == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_name == 'gradient_boosting':
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        model.fit(X_train, y_train)

        # Save the trained model
        self.models[model_name] = model

        # Save model to disk
        joblib.dump(model, os.path.join(self.output_dir, f"{model_name}.pkl"))

        print(f"{model_name} model trained and saved")
        return model

    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a trained model"""
        print(f"Evaluating {model_name} model...")

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Store metrics
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

        self.metrics[model_name] = metrics

        print(f"{model_name} evaluation metrics:")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  R²: {r2:.2f}")

        return metrics

    def get_feature_importance(self, model, feature_names, model_name):
        """Extract feature importance from a trained model"""
        print(f"Extracting feature importance for {model_name}...")

        # Check if model has feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            print(f"Model {model_name} does not have feature importance attributes")
            return None

        # Create a DataFrame of feature importances
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })

        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)

        # Save feature importance
        self.feature_importance[model_name] = importance_df

        # Save to CSV
        importance_df.to_csv(os.path.join(self.output_dir, f"{model_name}_importance.csv"), index=False)

        print(f"Feature importance extracted and saved for {model_name}")
        return importance_df

    def find_important_skills(self, importance_df, skill_list, top_n=10):
        """Find the most important skills from feature importance"""
        print(f"Finding top {top_n} important skills...")

        # Filter importance DataFrame to only include skills
        skill_importance = importance_df[importance_df['feature'].isin(skill_list)]

        # Get top N skills
        top_skills = skill_importance.head(top_n)

        print(f"Top {top_n} important skills:")
        for i, (_, row) in enumerate(top_skills.iterrows(), 1):
            print(f"{i}. {row['feature']} (Importance: {row['importance']:.4f})")

        return top_skills

    def train_and_evaluate(self, df, target_column='salary_annual'):
        """Complete model training and evaluation pipeline"""
        # Prepare data
        X_train, X_test, y_train, y_test, feature_names = self.prepare_data(df, target_column)

        # Train Model 1: Random Forest
        model1 = self.train_tree_model(X_train, y_train, 'random_forest')
        metrics1 = self.evaluate_model(model1, X_test, y_test, 'random_forest')
        importance1 = self.get_feature_importance(model1, feature_names, 'random_forest')

        # Train Model 2: Gradient Boosting
        model2 = self.train_tree_model(X_train, y_train, 'gradient_boosting')
        metrics2 = self.evaluate_model(model2, X_test, y_test, 'gradient_boosting')
        importance2 = self.get_feature_importance(model2, feature_names, 'gradient_boosting')

        # Extract skill list
        skill_list = [col for col in feature_names if col in self._get_common_skills()]

        # Find important skills
        if importance1 is not None:
            top_skills1 = self.find_important_skills(importance1, skill_list)

        if importance2 is not None:
            top_skills2 = self.find_important_skills(importance2, skill_list)

        print("Model training and evaluation complete")

        return {
            'models': self.models,
            'metrics': self.metrics,
            'importance': self.feature_importance,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names
        }

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