#!/usr/bin/env python3
"""
AI-Based Diabetes Prediction System - Data Preprocessing Module
================================================================

This module implements comprehensive data preprocessing for the Pima Indians Diabetes dataset.
It includes data loading, exploration, cleaning, feature engineering, and preparation for ML models.

Author: AI-Based Diabetes Prediction System Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DiabetesDataPreprocessor:
    """
    Comprehensive data preprocessing class for diabetes prediction dataset.
    """
    
    def __init__(self, data_path):
        """
        Initialize the preprocessor with dataset path.
        
        Args:
            data_path (str): Path to the diabetes dataset CSV file
        """
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
        ]
        
    def load_data(self):
        """
        Load the diabetes dataset and add column names.
        """
        print("Loading diabetes dataset...")
        self.data = pd.read_csv(self.data_path, header=None)
        self.data.columns = self.feature_names
        print(f"Dataset loaded successfully. Shape: {self.data.shape}")
        return self.data
    
    def explore_data(self):
        """
        Perform comprehensive exploratory data analysis.
        """
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Basic information
        print("\n1. Dataset Overview:")
        print(f"   - Shape: {self.data.shape}")
        print(f"   - Features: {len(self.data.columns)-1}")
        print(f"   - Samples: {len(self.data)}")
        
        # Data types and missing values
        print("\n2. Data Types and Missing Values:")
        print(self.data.info())
        
        # Statistical summary
        print("\n3. Statistical Summary:")
        print(self.data.describe())
        
        # Target variable distribution
        print("\n4. Target Variable Distribution:")
        outcome_counts = self.data['Outcome'].value_counts()
        print(f"   - Non-diabetic (0): {outcome_counts[0]} ({outcome_counts[0]/len(self.data)*100:.1f}%)")
        print(f"   - Diabetic (1): {outcome_counts[1]} ({outcome_counts[1]/len(self.data)*100:.1f}%)")
        
        # Check for zero values (which might indicate missing data)
        print("\n5. Zero Values Analysis (potential missing data):")
        zero_counts = {}
        for col in self.data.columns[:-1]:  # Exclude target variable
            zero_count = (self.data[col] == 0).sum()
            zero_percentage = (zero_count / len(self.data)) * 100
            zero_counts[col] = {'count': zero_count, 'percentage': zero_percentage}
            if zero_count > 0:
                print(f"   - {col}: {zero_count} zeros ({zero_percentage:.1f}%)")
        
        return zero_counts
    
    def visualize_data(self):
        """
        Create comprehensive data visualizations.
        """
        print("\n" + "="*60)
        print("DATA VISUALIZATION")
        print("="*60)
        
        # Set up the plotting area
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Target variable distribution
        plt.subplot(3, 4, 1)
        self.data['Outcome'].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
        plt.title('Target Variable Distribution')
        plt.xlabel('Outcome (0: Non-diabetic, 1: Diabetic)')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        
        # 2. Age distribution by outcome
        plt.subplot(3, 4, 2)
        self.data.boxplot(column='Age', by='Outcome', ax=plt.gca())
        plt.title('Age Distribution by Outcome')
        plt.suptitle('')  # Remove default title
        
        # 3. Glucose distribution by outcome
        plt.subplot(3, 4, 3)
        self.data.boxplot(column='Glucose', by='Outcome', ax=plt.gca())
        plt.title('Glucose Distribution by Outcome')
        plt.suptitle('')
        
        # 4. BMI distribution by outcome
        plt.subplot(3, 4, 4)
        self.data.boxplot(column='BMI', by='Outcome', ax=plt.gca())
        plt.title('BMI Distribution by Outcome')
        plt.suptitle('')
        
        # 5. Correlation heatmap
        plt.subplot(3, 4, 5)
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix')
        
        # 6. Pregnancies distribution
        plt.subplot(3, 4, 6)
        self.data['Pregnancies'].hist(bins=15, alpha=0.7, color='lightgreen')
        plt.title('Pregnancies Distribution')
        plt.xlabel('Number of Pregnancies')
        plt.ylabel('Frequency')
        
        # 7. Insulin distribution (log scale due to skewness)
        plt.subplot(3, 4, 7)
        insulin_non_zero = self.data[self.data['Insulin'] > 0]['Insulin']
        insulin_non_zero.hist(bins=20, alpha=0.7, color='orange')
        plt.title('Insulin Distribution (Non-zero values)')
        plt.xlabel('Insulin Level')
        plt.ylabel('Frequency')
        
        # 8. Blood Pressure distribution
        plt.subplot(3, 4, 8)
        bp_non_zero = self.data[self.data['BloodPressure'] > 0]['BloodPressure']
        bp_non_zero.hist(bins=20, alpha=0.7, color='purple')
        plt.title('Blood Pressure Distribution (Non-zero values)')
        plt.xlabel('Blood Pressure')
        plt.ylabel('Frequency')
        
        # 9. Pairplot for key features
        plt.subplot(3, 4, 9)
        key_features = ['Glucose', 'BMI', 'Age', 'Outcome']
        for i, outcome in enumerate([0, 1]):
            subset = self.data[self.data['Outcome'] == outcome]
            plt.scatter(subset['Glucose'], subset['BMI'], 
                       alpha=0.6, label=f'Outcome {outcome}')
        plt.xlabel('Glucose')
        plt.ylabel('BMI')
        plt.title('Glucose vs BMI by Outcome')
        plt.legend()
        
        # 10. Diabetes Pedigree Function distribution
        plt.subplot(3, 4, 10)
        self.data['DiabetesPedigreeFunction'].hist(bins=20, alpha=0.7, color='red')
        plt.title('Diabetes Pedigree Function Distribution')
        plt.xlabel('Diabetes Pedigree Function')
        plt.ylabel('Frequency')
        
        # 11. Skin Thickness distribution
        plt.subplot(3, 4, 11)
        skin_non_zero = self.data[self.data['SkinThickness'] > 0]['SkinThickness']
        skin_non_zero.hist(bins=20, alpha=0.7, color='brown')
        plt.title('Skin Thickness Distribution (Non-zero values)')
        plt.xlabel('Skin Thickness')
        plt.ylabel('Frequency')
        
        # 12. Feature importance visualization (correlation with target)
        plt.subplot(3, 4, 12)
        feature_corr = self.data.corr()['Outcome'].drop('Outcome').sort_values(key=abs, ascending=False)
        feature_corr.plot(kind='barh', color='teal')
        plt.title('Feature Correlation with Target')
        plt.xlabel('Correlation Coefficient')
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/data_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Data visualization completed. Saved as 'data_exploration.png'")
    
    def handle_missing_values(self):
        """
        Handle missing values represented as zeros in certain features.
        """
        print("\n" + "="*60)
        print("MISSING VALUE HANDLING")
        print("="*60)
        
        # Features that shouldn't have zero values
        zero_not_allowed = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        print("Replacing zero values with NaN for features where zero is not physiologically possible:")
        
        for feature in zero_not_allowed:
            zero_count = (self.data[feature] == 0).sum()
            if zero_count > 0:
                print(f"   - {feature}: {zero_count} zeros replaced with NaN")
                self.data[feature] = self.data[feature].replace(0, np.nan)
        
        # Display missing value statistics
        print("\nMissing value statistics after replacement:")
        missing_stats = self.data.isnull().sum()
        for feature, count in missing_stats.items():
            if count > 0:
                percentage = (count / len(self.data)) * 100
                print(f"   - {feature}: {count} missing ({percentage:.1f}%)")
        
        # Impute missing values using KNN imputation
        print("\nApplying KNN imputation for missing values...")
        
        # Separate features and target
        X = self.data.drop('Outcome', axis=1)
        y = self.data['Outcome']
        
        # Apply KNN imputation
        imputer = KNNImputer(n_neighbors=5)
        X_imputed = imputer.fit_transform(X)
        
        # Create new dataframe with imputed values
        X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)
        self.data = pd.concat([X_imputed_df, y], axis=1)
        
        print("Missing value imputation completed using KNN (k=5)")
        
        # Verify no missing values remain
        remaining_missing = self.data.isnull().sum().sum()
        print(f"Remaining missing values: {remaining_missing}")
        
        return self.data
    
    def detect_outliers(self):
        """
        Detect and analyze outliers using IQR method.
        """
        print("\n" + "="*60)
        print("OUTLIER DETECTION")
        print("="*60)
        
        outlier_info = {}
        features = self.data.columns[:-1]  # Exclude target variable
        
        for feature in features:
            Q1 = self.data[feature].quantile(0.25)
            Q3 = self.data[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.data[(self.data[feature] < lower_bound) | 
                               (self.data[feature] > upper_bound)]
            
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(self.data)) * 100
            
            outlier_info[feature] = {
                'count': outlier_count,
                'percentage': outlier_percentage,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            if outlier_count > 0:
                print(f"   - {feature}: {outlier_count} outliers ({outlier_percentage:.1f}%)")
                print(f"     Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        return outlier_info
    
    def feature_engineering(self):
        """
        Create new features and transform existing ones.
        """
        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)
        
        # Create new features
        print("Creating new features:")
        
        # 1. BMI categories
        self.data['BMI_Category'] = pd.cut(self.data['BMI'], 
                                          bins=[0, 18.5, 25, 30, float('inf')],
                                          labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        print("   - BMI_Category: Categorical BMI classification")
        
        # 2. Age groups
        self.data['Age_Group'] = pd.cut(self.data['Age'],
                                       bins=[0, 30, 40, 50, float('inf')],
                                       labels=['Young', 'Middle_Young', 'Middle_Old', 'Old'])
        print("   - Age_Group: Age group classification")
        
        # 3. Glucose categories
        self.data['Glucose_Category'] = pd.cut(self.data['Glucose'],
                                              bins=[0, 100, 125, float('inf')],
                                              labels=['Normal', 'Prediabetic', 'Diabetic'])
        print("   - Glucose_Category: Glucose level classification")
        
        # 4. Pregnancy risk
        self.data['High_Pregnancies'] = (self.data['Pregnancies'] >= 5).astype(int)
        print("   - High_Pregnancies: Binary indicator for high pregnancy count")
        
        # 5. Insulin resistance indicator
        self.data['Insulin_Resistance'] = ((self.data['Insulin'] > 166) | 
                                          (self.data['Glucose'] > 140)).astype(int)
        print("   - Insulin_Resistance: Binary indicator for insulin resistance")
        
        # 6. Risk score (composite feature)
        # Normalize features for risk score calculation
        glucose_norm = (self.data['Glucose'] - self.data['Glucose'].min()) / (self.data['Glucose'].max() - self.data['Glucose'].min())
        bmi_norm = (self.data['BMI'] - self.data['BMI'].min()) / (self.data['BMI'].max() - self.data['BMI'].min())
        age_norm = (self.data['Age'] - self.data['Age'].min()) / (self.data['Age'].max() - self.data['Age'].min())
        
        self.data['Risk_Score'] = (0.4 * glucose_norm + 0.3 * bmi_norm + 0.2 * age_norm + 
                                  0.1 * self.data['DiabetesPedigreeFunction'])
        print("   - Risk_Score: Composite risk score based on key features")
        
        # Convert categorical features to dummy variables
        categorical_features = ['BMI_Category', 'Age_Group', 'Glucose_Category']
        self.data = pd.get_dummies(self.data, columns=categorical_features, prefix=categorical_features)
        
        print(f"\nFeature engineering completed. New shape: {self.data.shape}")
        return self.data
    
    def prepare_for_modeling(self, test_size=0.2, random_state=42):
        """
        Prepare data for machine learning modeling.
        """
        print("\n" + "="*60)
        print("DATA PREPARATION FOR MODELING")
        print("="*60)
        
        # Separate features and target
        X = self.data.drop('Outcome', axis=1)
        y = self.data['Outcome']
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTrain set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        # Check class distribution
        train_dist = self.y_train.value_counts(normalize=True)
        test_dist = self.y_test.value_counts(normalize=True)
        
        print(f"\nTrain set class distribution:")
        print(f"   - Non-diabetic: {train_dist[0]:.3f}")
        print(f"   - Diabetic: {train_dist[1]:.3f}")
        
        print(f"\nTest set class distribution:")
        print(f"   - Non-diabetic: {test_dist[0]:.3f}")
        print(f"   - Diabetic: {test_dist[1]:.3f}")
        
        # Scale the features
        print("\nApplying feature scaling (StandardScaler)...")
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Convert back to DataFrames for easier handling
        self.X_train_scaled = pd.DataFrame(self.X_train_scaled, columns=X.columns)
        self.X_test_scaled = pd.DataFrame(self.X_test_scaled, columns=X.columns)
        
        print("Data preparation completed successfully!")
        
        return (self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test)
    
    def save_processed_data(self):
        """
        Save processed data for future use.
        """
        print("\n" + "="*60)
        print("SAVING PROCESSED DATA")
        print("="*60)
        
        # Save the complete processed dataset
        self.data.to_csv('/home/ubuntu/processed_diabetes_data.csv', index=False)
        print("Complete processed dataset saved as 'processed_diabetes_data.csv'")
        
        # Save train/test splits
        self.X_train_scaled.to_csv('/home/ubuntu/X_train.csv', index=False)
        self.X_test_scaled.to_csv('/home/ubuntu/X_test.csv', index=False)
        self.y_train.to_csv('/home/ubuntu/y_train.csv', index=False)
        self.y_test.to_csv('/home/ubuntu/y_test.csv', index=False)
        
        print("Train/test splits saved as separate CSV files")
        
        # Save feature names
        feature_names = list(self.X_train_scaled.columns)
        with open('/home/ubuntu/feature_names.txt', 'w') as f:
            for feature in feature_names:
                f.write(f"{feature}\n")
        
        print(f"Feature names saved. Total features: {len(feature_names)}")
        
        return feature_names

def main():
    """
    Main function to run the complete data preprocessing pipeline.
    """
    print("AI-BASED DIABETES PREDICTION SYSTEM")
    print("Data Preprocessing Pipeline")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = DiabetesDataPreprocessor('/home/ubuntu/pima_diabetes.csv')
    
    # Run preprocessing pipeline
    try:
        # Load data
        data = preprocessor.load_data()
        
        # Explore data
        zero_counts = preprocessor.explore_data()
        
        # Visualize data
        preprocessor.visualize_data()
        
        # Handle missing values
        preprocessor.handle_missing_values()
        
        # Detect outliers
        outlier_info = preprocessor.detect_outliers()
        
        # Feature engineering
        preprocessor.feature_engineering()
        
        # Prepare for modeling
        X_train, X_test, y_train, y_test = preprocessor.prepare_for_modeling()
        
        # Save processed data
        feature_names = preprocessor.save_processed_data()
        
        print("\n" + "="*60)
        print("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Final dataset shape: {preprocessor.data.shape}")
        print(f"Number of features: {len(feature_names)}")
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        return preprocessor
        
    except Exception as e:
        print(f"Error in preprocessing pipeline: {str(e)}")
        return None

if __name__ == "__main__":
    preprocessor = main()

