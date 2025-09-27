#!/usr/bin/env python3
"""
AI-Based Diabetes Prediction System - Machine Learning Model Development
========================================================================

This module implements comprehensive machine learning model development for diabetes prediction.
It includes multiple algorithms, hyperparameter tuning, ensemble methods, and model evaluation.

Author: AI-Based Diabetes Prediction System Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report, roc_curve)
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DiabetesMLModels:
    """
    Comprehensive machine learning model development class for diabetes prediction.
    """
    
    def __init__(self):
        """
        Initialize the ML models class.
        """
        self.models = {}
        self.best_models = {}
        self.model_scores = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def load_processed_data(self):
        """
        Load the preprocessed data for model training.
        """
        print("Loading preprocessed data...")
        
        # Load train/test splits
        self.X_train = pd.read_csv('/home/ubuntu/X_train.csv')
        self.X_test = pd.read_csv('/home/ubuntu/X_test.csv')
        self.y_train = pd.read_csv('/home/ubuntu/y_train.csv').squeeze()
        self.y_test = pd.read_csv('/home/ubuntu/y_test.csv').squeeze()
        
        # Load feature names
        with open('/home/ubuntu/feature_names.txt', 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
        
        print(f"Data loaded successfully:")
        print(f"   - Training samples: {self.X_train.shape[0]}")
        print(f"   - Testing samples: {self.X_test.shape[0]}")
        print(f"   - Features: {self.X_train.shape[1]}")
        
        return True
    
    def initialize_models(self):
        """
        Initialize all machine learning models with default parameters.
        """
        print("\n" + "="*60)
        print("INITIALIZING MACHINE LEARNING MODELS")
        print("="*60)
        
        self.models = {
            'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random_Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'SVM': SVC(random_state=42, probability=True),
            'Naive_Bayes': GaussianNB(),
            'KNN': KNeighborsClassifier(),
            'Decision_Tree': DecisionTreeClassifier(random_state=42),
            'Neural_Network': MLPClassifier(random_state=42, max_iter=1000)
        }
        
        print(f"Initialized {len(self.models)} machine learning models:")
        for model_name in self.models.keys():
            print(f"   - {model_name}")
        
        return self.models
    
    def train_baseline_models(self):
        """
        Train all models with default parameters to establish baselines.
        """
        print("\n" + "="*60)
        print("TRAINING BASELINE MODELS")
        print("="*60)
        
        baseline_scores = {}
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            
            try:
                # Train the model
                model.fit(self.X_train, self.y_train)
                
                # Make predictions
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred)
                recall = recall_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)
                auc = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else 0
                
                # Cross-validation score
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                baseline_scores[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc_roc': auc,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std
                }
                
                print(f"   Accuracy: {accuracy:.4f}")
                print(f"   Precision: {precision:.4f}")
                print(f"   Recall: {recall:.4f}")
                print(f"   F1-Score: {f1:.4f}")
                print(f"   AUC-ROC: {auc:.4f}")
                print(f"   CV Score: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
                
            except Exception as e:
                print(f"   Error training {model_name}: {str(e)}")
                baseline_scores[model_name] = None
        
        self.baseline_scores = baseline_scores
        return baseline_scores
    
    def hyperparameter_tuning(self):
        """
        Perform hyperparameter tuning for selected models.
        """
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING")
        print("="*60)
        
        # Define parameter grids for each model
        param_grids = {
            'Logistic_Regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'Random_Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
        
        tuned_models = {}
        
        for model_name in ['Logistic_Regression', 'Random_Forest', 'XGBoost', 'SVM', 'KNN']:
            print(f"\nTuning {model_name}...")
            
            try:
                # Get the base model
                base_model = self.models[model_name]
                
                # Perform grid search
                grid_search = GridSearchCV(
                    base_model, 
                    param_grids[model_name],
                    cv=5,
                    scoring='f1',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(self.X_train, self.y_train)
                
                # Store the best model
                tuned_models[model_name] = grid_search.best_estimator_
                
                print(f"   Best parameters: {grid_search.best_params_}")
                print(f"   Best CV score: {grid_search.best_score_:.4f}")
                
                # Evaluate on test set
                y_pred = grid_search.best_estimator_.predict(self.X_test)
                test_f1 = f1_score(self.y_test, y_pred)
                print(f"   Test F1-score: {test_f1:.4f}")
                
            except Exception as e:
                print(f"   Error tuning {model_name}: {str(e)}")
                tuned_models[model_name] = self.models[model_name]
        
        # Update models with tuned versions
        self.models.update(tuned_models)
        self.best_models = tuned_models
        
        return tuned_models
    
    def create_ensemble_models(self):
        """
        Create ensemble models using voting and stacking.
        """
        print("\n" + "="*60)
        print("CREATING ENSEMBLE MODELS")
        print("="*60)
        
        # Select best performing models for ensemble
        best_models_list = [
            ('lr', self.models['Logistic_Regression']),
            ('rf', self.models['Random_Forest']),
            ('xgb', self.models['XGBoost']),
            ('svm', self.models['SVM'])
        ]
        
        # Voting Classifier (Hard Voting)
        print("Creating Voting Classifier (Hard Voting)...")
        voting_hard = VotingClassifier(
            estimators=best_models_list,
            voting='hard'
        )
        
        # Voting Classifier (Soft Voting)
        print("Creating Voting Classifier (Soft Voting)...")
        voting_soft = VotingClassifier(
            estimators=best_models_list,
            voting='soft'
        )
        
        # Stacking Classifier
        print("Creating Stacking Classifier...")
        stacking = StackingClassifier(
            estimators=best_models_list,
            final_estimator=LogisticRegression(),
            cv=5
        )
        
        # Train ensemble models
        ensemble_models = {
            'Voting_Hard': voting_hard,
            'Voting_Soft': voting_soft,
            'Stacking': stacking
        }
        
        ensemble_scores = {}
        
        for ensemble_name, ensemble_model in ensemble_models.items():
            print(f"\nTraining {ensemble_name}...")
            
            try:
                # Train the ensemble
                ensemble_model.fit(self.X_train, self.y_train)
                
                # Make predictions
                y_pred = ensemble_model.predict(self.X_test)
                y_pred_proba = ensemble_model.predict_proba(self.X_test)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred)
                recall = recall_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)
                auc = roc_auc_score(self.y_test, y_pred_proba)
                
                ensemble_scores[ensemble_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc_roc': auc
                }
                
                print(f"   Accuracy: {accuracy:.4f}")
                print(f"   Precision: {precision:.4f}")
                print(f"   Recall: {recall:.4f}")
                print(f"   F1-Score: {f1:.4f}")
                print(f"   AUC-ROC: {auc:.4f}")
                
            except Exception as e:
                print(f"   Error training {ensemble_name}: {str(e)}")
        
        # Add ensemble models to main models dictionary
        self.models.update(ensemble_models)
        self.ensemble_scores = ensemble_scores
        
        return ensemble_models, ensemble_scores
    
    def evaluate_all_models(self):
        """
        Comprehensive evaluation of all trained models.
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*60)
        
        all_scores = {}
        
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            
            try:
                # Make predictions
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate comprehensive metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred)
                recall = recall_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)
                auc = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else 0
                
                # Confusion matrix
                cm = confusion_matrix(self.y_test, y_pred)
                tn, fp, fn, tp = cm.ravel()
                
                # Additional metrics
                specificity = tn / (tn + fp)
                sensitivity = tp / (tp + fn)  # Same as recall
                npv = tn / (tn + fn)  # Negative Predictive Value
                ppv = tp / (tp + fp)  # Positive Predictive Value (same as precision)
                
                all_scores[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc_roc': auc,
                    'specificity': specificity,
                    'sensitivity': sensitivity,
                    'npv': npv,
                    'ppv': ppv,
                    'confusion_matrix': cm,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"   Accuracy: {accuracy:.4f}")
                print(f"   Precision: {precision:.4f}")
                print(f"   Recall: {recall:.4f}")
                print(f"   F1-Score: {f1:.4f}")
                print(f"   AUC-ROC: {auc:.4f}")
                print(f"   Specificity: {specificity:.4f}")
                
            except Exception as e:
                print(f"   Error evaluating {model_name}: {str(e)}")
                all_scores[model_name] = None
        
        self.model_scores = all_scores
        return all_scores
    
    def save_models(self):
        """
        Save all trained models for future use.
        """
        print("\n" + "="*60)
        print("SAVING TRAINED MODELS")
        print("="*60)
        
        # Create models directory
        import os
        os.makedirs('/home/ubuntu/models', exist_ok=True)
        
        saved_models = []
        
        for model_name, model in self.models.items():
            try:
                model_path = f'/home/ubuntu/models/{model_name}.joblib'
                joblib.dump(model, model_path)
                saved_models.append(model_name)
                print(f"   Saved {model_name}")
            except Exception as e:
                print(f"   Error saving {model_name}: {str(e)}")
        
        # Save model scores
        scores_df = pd.DataFrame(self.model_scores).T
        scores_df.to_csv('/home/ubuntu/model_scores.csv')
        print(f"   Saved model scores to 'model_scores.csv'")
        
        print(f"\nTotal models saved: {len(saved_models)}")
        return saved_models
    
    def generate_model_comparison(self):
        """
        Generate comprehensive model comparison report.
        """
        print("\n" + "="*60)
        print("GENERATING MODEL COMPARISON REPORT")
        print("="*60)
        
        # Create comparison DataFrame
        comparison_data = []
        
        for model_name, scores in self.model_scores.items():
            if scores is not None:
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': scores['accuracy'],
                    'Precision': scores['precision'],
                    'Recall': scores['recall'],
                    'F1-Score': scores['f1_score'],
                    'AUC-ROC': scores['auc_roc'],
                    'Specificity': scores['specificity']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        # Save comparison report
        comparison_df.to_csv('/home/ubuntu/model_comparison.csv', index=False)
        
        print("Model Performance Ranking (by F1-Score):")
        print("="*50)
        for idx, row in comparison_df.iterrows():
            print(f"{idx+1:2d}. {row['Model']:20s} - F1: {row['F1-Score']:.4f}, AUC: {row['AUC-ROC']:.4f}")
        
        # Identify best model
        best_model_name = comparison_df.iloc[0]['Model']
        best_f1_score = comparison_df.iloc[0]['F1-Score']
        
        print(f"\nBest performing model: {best_model_name}")
        print(f"Best F1-Score: {best_f1_score:.4f}")
        
        return comparison_df, best_model_name

def main():
    """
    Main function to run the complete ML model development pipeline.
    """
    print("AI-BASED DIABETES PREDICTION SYSTEM")
    print("Machine Learning Model Development Pipeline")
    print("="*60)
    
    # Initialize ML models class
    ml_models = DiabetesMLModels()
    
    try:
        # Load preprocessed data
        ml_models.load_processed_data()
        
        # Initialize models
        ml_models.initialize_models()
        
        # Train baseline models
        baseline_scores = ml_models.train_baseline_models()
        
        # Hyperparameter tuning
        tuned_models = ml_models.hyperparameter_tuning()
        
        # Create ensemble models
        ensemble_models, ensemble_scores = ml_models.create_ensemble_models()
        
        # Comprehensive evaluation
        all_scores = ml_models.evaluate_all_models()
        
        # Save models
        saved_models = ml_models.save_models()
        
        # Generate comparison report
        comparison_df, best_model = ml_models.generate_model_comparison()
        
        print("\n" + "="*60)
        print("MODEL DEVELOPMENT COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Total models trained: {len(ml_models.models)}")
        print(f"Best performing model: {best_model}")
        print(f"Models saved: {len(saved_models)}")
        
        return ml_models
        
    except Exception as e:
        print(f"Error in ML model development pipeline: {str(e)}")
        return None

if __name__ == "__main__":
    ml_models = main()

