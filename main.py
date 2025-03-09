import argparse
import os
import pandas as pd
from datetime import datetime
from preprocessing import DataPreprocessor
from feature_eng import FeatureEngineer
from model import ModelTrainer
from visualization import DataVisualizer

def main():
    parser = argparse.ArgumentParser(description='Job Market Analysis for Salary Prediction and Skill Identification')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess data')
    parser.add_argument('--engineer', action='store_true', help='Engineer features')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    parser.add_argument('--data', type=str, default='job_data.csv', help='Path to the data file')
    parser.add_argument('--output', type=str, default='output', help='Output directory')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # If no specific step is specified, run all steps
    if not any([args.preprocess, args.engineer, args.train, args.visualize]):
        args.all = True

    # Step 1: Preprocess data
    if args.preprocess or args.all:
        print("\nStep 1: Preprocessing data...")
        preprocessor = DataPreprocessor()

        # Load data
        df = preprocessor.load_data(args.data)

        # Preprocess data
        df_processed = preprocessor.process_data(args.data)

        # Save preprocessed data
        preprocessed_file = os.path.join(args.output, 'preprocessed_data.csv')
        df_processed.to_csv(preprocessed_file, index=False)
        print(f"Preprocessed data saved to {preprocessed_file}")

    # Step 2: Engineer features
    if args.engineer or args.all:
        print("\nStep 2: Engineering features...")
        engineer = FeatureEngineer()

        # Load preprocessed data if not already loaded
        if 'df_processed' not in locals():
            preprocessed_file = os.path.join(args.output, 'preprocessed_data.csv')
            if os.path.exists(preprocessed_file):
                df_processed = pd.read_csv(preprocessed_file)
            else:
                print("Preprocessed data not found. Please run preprocessing step first.")
                return

        # Engineer features
        df_engineered = engineer.engineer_features(df_processed)

        # Save engineered data
        engineered_file = os.path.join(args.output, 'engineered_data.csv')
        df_engineered.to_csv(engineered_file, index=False)
        print(f"Engineered data saved to {engineered_file}")

    # Step 3: Train models
    if args.train or args.all:
        print("\nStep 3: Training models...")
        trainer = ModelTrainer(output_dir=os.path.join(args.output, 'models'))

        # Load engineered data if not already loaded
        if 'df_engineered' not in locals():
            engineered_file = os.path.join(args.output, 'engineered_data.csv')
            if os.path.exists(engineered_file):
                df_engineered = pd.read_csv(engineered_file)
            else:
                print("Engineered data not found. Please run feature engineering step first.")
                return

        # Train and evaluate models
        results = trainer.train_and_evaluate(df_engineered)

        # Save training results
        results_file = os.path.join(args.output, 'model_results.txt')
        with open(results_file, 'w') as f:
            f.write("Model Training Results\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for model_name, metrics in results['metrics'].items():
                f.write(f"Model: {model_name}\n")
                f.write(f"  RMSE: {metrics['rmse']:.2f}\n")
                f.write(f"  MAE: {metrics['mae']:.2f}\n")
                f.write(f"  R²: {metrics['r2']:.2f}\n\n")

        print(f"Model training results saved to {results_file}")

    # Step 4: Create visualizations
    if args.visualize or args.all:
        print("\nStep 4: Creating visualizations...")
        visualizer = DataVisualizer(output_dir=os.path.join(args.output, 'visualizations'))

        # Load engineered data if not already loaded
        if 'df_engineered' not in locals():
            engineered_file = os.path.join(args.output, 'engineered_data.csv')
            if os.path.exists(engineered_file):
                df_engineered = pd.read_csv(engineered_file)
            else:
                print("Engineered data not found. Please run feature engineering step first.")
                return

        # Load trained models if not already loaded
        if 'results' not in locals():
            model_dir = os.path.join(args.output, 'models')
            if os.path.exists(os.path.join(model_dir, 'random_forest.pkl')) and os.path.exists(os.path.join(model_dir, 'gradient_boosting.pkl')):
                import joblib
                model1 = joblib.load(os.path.join(model_dir, 'random_forest.pkl'))
                model2 = joblib.load(os.path.join(model_dir, 'gradient_boosting.pkl'))

                # Prepare data for visualization
                X_train, X_test, y_train, y_test, feature_names = ModelTrainer().prepare_data(df_engineered)

                results = {
                    'models': {'random_forest': model1, 'gradient_boosting': model2},
                    'X_test': X_test,
                    'y_test': y_test,
                    'feature_names': feature_names
                }
            else:
                print("Trained models not found. Will create visualizations without model information.")
                results = None

        # Create visualizations
        if results:
            visualizer.visualize_data(
                df_engineered,
                model1=results['models']['random_forest'],
                model2=results['models']['gradient_boosting'],
                X=results['X_test'],
                y=results['y_test'],
                feature_names=results['feature_names']
            )
        else:
            visualizer.visualize_data(df_engineered)

        print("Visualizations created successfully")

    print("\nJob Market Analysis completed successfully!")

if __name__ == "__main__":
    main()