#!/usr/bin/env python3
"""
AI-Based Diabetes Prediction System - Model Testing and Performance Evaluation
==============================================================================

This module implements comprehensive model testing, validation, and performance evaluation
for the diabetes prediction system. It includes error analysis, robustness testing,
and detailed performance metrics calculation.

Author: AI-Based Diabetes Prediction System Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report, 
                           roc_curve, precision_recall_curve, average_precision_score)
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelTestingEvaluation:
    """
    Comprehensive model testing and evaluation class.
    """
    
    def __init__(self):
        """
        Initialize the testing and evaluation class.
        """
        self.models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model_scores = None
        self.test_results = {}
        
    def load_data_and_models(self):
        """
        Load preprocessed data and trained models.
        """
        print("Loading data and trained models...")
        
        # Load data
        self.X_train = pd.read_csv('/home/ubuntu/X_train.csv')
        self.X_test = pd.read_csv('/home/ubuntu/X_test.csv')
        self.y_train = pd.read_csv('/home/ubuntu/y_train.csv').squeeze()
        self.y_test = pd.read_csv('/home/ubuntu/y_test.csv').squeeze()
        
        # Load model scores
        self.model_scores = pd.read_csv('/home/ubuntu/model_scores.csv', index_col=0)
        
        # Load trained models
        import os
        model_dir = '/home/ubuntu/models'
        if os.path.exists(model_dir):
            for model_file in os.listdir(model_dir):
                if model_file.endswith('.joblib'):
                    model_name = model_file.replace('.joblib', '')
                    try:
                        model_path = os.path.join(model_dir, model_file)
                        self.models[model_name] = joblib.load(model_path)
                        print(f"   Loaded {model_name}")
                    except Exception as e:
                        print(f"   Error loading {model_name}: {str(e)}")
        
        print(f"Data loaded: {self.X_train.shape[0]} train, {self.X_test.shape[0]} test samples")
        print(f"Models loaded: {len(self.models)}")
        
        return True
    
    def perform_cross_validation_testing(self):
        """
        Perform comprehensive cross-validation testing.
        """
        print("\n" + "="*60)
        print("CROSS-VALIDATION TESTING")
        print("="*60)
        
        cv_results = {}
        
        for model_name, model in self.models.items():
            print(f"\nTesting {model_name} with cross-validation...")
            
            try:
                # Perform 5-fold cross-validation with multiple metrics
                cv_accuracy = cross_val_score(model, self.X_train, self.y_train, 
                                            cv=5, scoring='accuracy')
                cv_precision = cross_val_score(model, self.X_train, self.y_train, 
                                             cv=5, scoring='precision')
                cv_recall = cross_val_score(model, self.X_train, self.y_train, 
                                          cv=5, scoring='recall')
                cv_f1 = cross_val_score(model, self.X_train, self.y_train, 
                                       cv=5, scoring='f1')
                cv_roc_auc = cross_val_score(model, self.X_train, self.y_train, 
                                           cv=5, scoring='roc_auc')
                
                cv_results[model_name] = {
                    'accuracy_mean': cv_accuracy.mean(),
                    'accuracy_std': cv_accuracy.std(),
                    'precision_mean': cv_precision.mean(),
                    'precision_std': cv_precision.std(),
                    'recall_mean': cv_recall.mean(),
                    'recall_std': cv_recall.std(),
                    'f1_mean': cv_f1.mean(),
                    'f1_std': cv_f1.std(),
                    'roc_auc_mean': cv_roc_auc.mean(),
                    'roc_auc_std': cv_roc_auc.std()
                }
                
                print(f"   Accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std()*2:.4f})")
                print(f"   Precision: {cv_precision.mean():.4f} (+/- {cv_precision.std()*2:.4f})")
                print(f"   Recall: {cv_recall.mean():.4f} (+/- {cv_recall.std()*2:.4f})")
                print(f"   F1-Score: {cv_f1.mean():.4f} (+/- {cv_f1.std()*2:.4f})")
                print(f"   ROC-AUC: {cv_roc_auc.mean():.4f} (+/- {cv_roc_auc.std()*2:.4f})")
                
            except Exception as e:
                print(f"   Error in cross-validation for {model_name}: {str(e)}")
                cv_results[model_name] = None
        
        # Save cross-validation results
        cv_df = pd.DataFrame(cv_results).T
        cv_df.to_csv('/home/ubuntu/cross_validation_results.csv')
        
        self.cv_results = cv_results
        return cv_results
    
    def perform_robustness_testing(self):
        """
        Test model robustness with data perturbations.
        """
        print("\n" + "="*60)
        print("ROBUSTNESS TESTING")
        print("="*60)
        
        robustness_results = {}
        
        # Test with different noise levels
        noise_levels = [0.01, 0.05, 0.1, 0.2]
        
        for model_name, model in self.models.items():
            print(f"\nTesting robustness of {model_name}...")
            
            try:
                model_robustness = {}
                
                # Original performance
                y_pred_original = model.predict(self.X_test)
                original_accuracy = accuracy_score(self.y_test, y_pred_original)
                model_robustness['original_accuracy'] = original_accuracy
                
                # Test with noise
                for noise_level in noise_levels:
                    # Add Gaussian noise to test data
                    X_test_noisy = self.X_test + np.random.normal(0, noise_level, self.X_test.shape)
                    
                    # Make predictions on noisy data
                    y_pred_noisy = model.predict(X_test_noisy)
                    noisy_accuracy = accuracy_score(self.y_test, y_pred_noisy)
                    
                    # Calculate robustness score (how much performance degrades)
                    robustness_score = noisy_accuracy / original_accuracy
                    model_robustness[f'noise_{noise_level}_accuracy'] = noisy_accuracy
                    model_robustness[f'noise_{noise_level}_robustness'] = robustness_score
                
                robustness_results[model_name] = model_robustness
                
                print(f"   Original accuracy: {original_accuracy:.4f}")
                for noise_level in noise_levels:
                    robustness_score = model_robustness[f'noise_{noise_level}_robustness']
                    print(f"   Noise {noise_level}: {robustness_score:.4f} robustness")
                
            except Exception as e:
                print(f"   Error in robustness testing for {model_name}: {str(e)}")
                robustness_results[model_name] = None
        
        # Save robustness results
        robustness_df = pd.DataFrame(robustness_results).T
        robustness_df.to_csv('/home/ubuntu/robustness_test_results.csv')
        
        self.robustness_results = robustness_results
        return robustness_results
    
    def perform_error_analysis(self):
        """
        Perform detailed error analysis for all models.
        """
        print("\n" + "="*60)
        print("ERROR ANALYSIS")
        print("="*60)
        
        error_analysis = {}
        
        for model_name, model in self.models.items():
            print(f"\nAnalyzing errors for {model_name}...")
            
            try:
                # Make predictions
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Identify different types of errors
                true_positives = ((self.y_test == 1) & (y_pred == 1)).sum()
                true_negatives = ((self.y_test == 0) & (y_pred == 0)).sum()
                false_positives = ((self.y_test == 0) & (y_pred == 1)).sum()
                false_negatives = ((self.y_test == 1) & (y_pred == 0)).sum()
                
                # Calculate error rates
                total_samples = len(self.y_test)
                fp_rate = false_positives / total_samples
                fn_rate = false_negatives / total_samples
                
                # Analyze misclassified samples
                misclassified_indices = np.where(self.y_test != y_pred)[0]
                misclassified_features = self.X_test.iloc[misclassified_indices]
                
                # Calculate feature statistics for misclassified samples
                feature_stats = {}
                for feature in self.X_test.columns:
                    feature_stats[feature] = {
                        'mean_misclassified': misclassified_features[feature].mean(),
                        'mean_all': self.X_test[feature].mean(),
                        'std_misclassified': misclassified_features[feature].std(),
                        'std_all': self.X_test[feature].std()
                    }
                
                error_analysis[model_name] = {
                    'true_positives': true_positives,
                    'true_negatives': true_negatives,
                    'false_positives': false_positives,
                    'false_negatives': false_negatives,
                    'fp_rate': fp_rate,
                    'fn_rate': fn_rate,
                    'misclassified_count': len(misclassified_indices),
                    'misclassified_percentage': len(misclassified_indices) / total_samples * 100,
                    'feature_stats': feature_stats
                }
                
                print(f"   True Positives: {true_positives}")
                print(f"   True Negatives: {true_negatives}")
                print(f"   False Positives: {false_positives} ({fp_rate:.4f} rate)")
                print(f"   False Negatives: {false_negatives} ({fn_rate:.4f} rate)")
                print(f"   Misclassified: {len(misclassified_indices)} ({len(misclassified_indices)/total_samples*100:.2f}%)")
                
            except Exception as e:
                print(f"   Error in error analysis for {model_name}: {str(e)}")
                error_analysis[model_name] = None
        
        self.error_analysis = error_analysis
        return error_analysis
    
    def calculate_clinical_metrics(self):
        """
        Calculate clinical performance metrics relevant to diabetes prediction.
        """
        print("\n" + "="*60)
        print("CLINICAL PERFORMANCE METRICS")
        print("="*60)
        
        clinical_metrics = {}
        
        for model_name, model in self.models.items():
            print(f"\nCalculating clinical metrics for {model_name}...")
            
            try:
                # Make predictions
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Confusion matrix
                cm = confusion_matrix(self.y_test, y_pred)
                tn, fp, fn, tp = cm.ravel()
                
                # Clinical metrics
                sensitivity = tp / (tp + fn)  # True Positive Rate (Recall)
                specificity = tn / (tn + fp)  # True Negative Rate
                ppv = tp / (tp + fp)  # Positive Predictive Value (Precision)
                npv = tn / (tn + fn)  # Negative Predictive Value
                
                # Likelihood ratios
                lr_positive = sensitivity / (1 - specificity) if specificity != 1 else float('inf')
                lr_negative = (1 - sensitivity) / specificity if specificity != 0 else float('inf')
                
                # Diagnostic odds ratio
                dor = lr_positive / lr_negative if lr_negative != 0 else float('inf')
                
                # Youden's J statistic
                youden_j = sensitivity + specificity - 1
                
                # Number needed to diagnose
                nnd = 1 / youden_j if youden_j != 0 else float('inf')
                
                clinical_metrics[model_name] = {
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'ppv': ppv,
                    'npv': npv,
                    'lr_positive': lr_positive,
                    'lr_negative': lr_negative,
                    'dor': dor,
                    'youden_j': youden_j,
                    'nnd': nnd
                }
                
                print(f"   Sensitivity (Recall): {sensitivity:.4f}")
                print(f"   Specificity: {specificity:.4f}")
                print(f"   PPV (Precision): {ppv:.4f}")
                print(f"   NPV: {npv:.4f}")
                print(f"   LR+: {lr_positive:.4f}")
                print(f"   LR-: {lr_negative:.4f}")
                print(f"   Youden's J: {youden_j:.4f}")
                
            except Exception as e:
                print(f"   Error calculating clinical metrics for {model_name}: {str(e)}")
                clinical_metrics[model_name] = None
        
        # Save clinical metrics
        clinical_df = pd.DataFrame(clinical_metrics).T
        clinical_df.to_csv('/home/ubuntu/clinical_metrics.csv')
        
        self.clinical_metrics = clinical_metrics
        return clinical_metrics
    
    def generate_learning_curves(self):
        """
        Generate learning curves for model performance analysis.
        """
        print("\n" + "="*60)
        print("GENERATING LEARNING CURVES")
        print("="*60)
        
        # Select top 4 models for learning curves
        top_models = ['Naive_Bayes', 'KNN', 'Logistic_Regression', 'Random_Forest']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, model_name in enumerate(top_models):
            if model_name in self.models:
                print(f"Generating learning curve for {model_name}...")
                
                try:
                    model = self.models[model_name]
                    
                    # Generate learning curve
                    train_sizes, train_scores, val_scores = learning_curve(
                        model, self.X_train, self.y_train,
                        cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
                        scoring='f1'
                    )
                    
                    # Calculate mean and std
                    train_mean = np.mean(train_scores, axis=1)
                    train_std = np.std(train_scores, axis=1)
                    val_mean = np.mean(val_scores, axis=1)
                    val_std = np.std(val_scores, axis=1)
                    
                    # Plot learning curve
                    axes[idx].plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
                    axes[idx].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
                    
                    axes[idx].plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
                    axes[idx].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
                    
                    axes[idx].set_title(f'Learning Curve - {model_name}')
                    axes[idx].set_xlabel('Training Set Size')
                    axes[idx].set_ylabel('F1-Score')
                    axes[idx].legend()
                    axes[idx].grid(True)
                    
                except Exception as e:
                    print(f"   Error generating learning curve for {model_name}: {str(e)}")
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/learning_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Learning curves saved as 'learning_curves.png'")
    
    def identify_and_fix_issues(self):
        """
        Identify potential issues and suggest fixes.
        """
        print("\n" + "="*60)
        print("ISSUE IDENTIFICATION AND FIXES")
        print("="*60)
        
        issues_and_fixes = {}
        
        for model_name in self.models.keys():
            print(f"\nAnalyzing {model_name} for potential issues...")
            
            issues = []
            fixes = []
            
            # Check model scores
            if model_name in self.model_scores.index:
                scores = self.model_scores.loc[model_name]
                
                # Issue 1: Low precision (high false positive rate)
                if scores['precision'] < 0.7:
                    issues.append(f"Low precision ({scores['precision']:.3f}) - High false positive rate")
                    fixes.append("Consider adjusting decision threshold, feature selection, or class balancing")
                
                # Issue 2: Low recall (high false negative rate)
                if scores['recall'] < 0.7:
                    issues.append(f"Low recall ({scores['recall']:.3f}) - High false negative rate")
                    fixes.append("Consider ensemble methods, feature engineering, or threshold tuning")
                
                # Issue 3: Large gap between precision and recall
                if abs(scores['precision'] - scores['recall']) > 0.2:
                    issues.append(f"Imbalanced precision-recall (P:{scores['precision']:.3f}, R:{scores['recall']:.3f})")
                    fixes.append("Consider cost-sensitive learning or SMOTE for class balancing")
                
                # Issue 4: Low AUC-ROC
                if scores['auc_roc'] < 0.8:
                    issues.append(f"Low AUC-ROC ({scores['auc_roc']:.3f}) - Poor discrimination ability")
                    fixes.append("Consider feature engineering, model complexity adjustment, or ensemble methods")
            
            # Check cross-validation results for overfitting
            if hasattr(self, 'cv_results') and model_name in self.cv_results:
                cv_data = self.cv_results[model_name]
                if cv_data and cv_data['f1_std'] > 0.1:
                    issues.append(f"High CV variance ({cv_data['f1_std']:.3f}) - Potential overfitting")
                    fixes.append("Consider regularization, cross-validation, or reducing model complexity")
            
            # Check robustness
            if hasattr(self, 'robustness_results') and model_name in self.robustness_results:
                rob_data = self.robustness_results[model_name]
                if rob_data and rob_data.get('noise_0.1_robustness', 1) < 0.9:
                    issues.append("Low robustness to noise - Model may be overfitting")
                    fixes.append("Consider data augmentation, regularization, or ensemble methods")
            
            issues_and_fixes[model_name] = {
                'issues': issues,
                'fixes': fixes,
                'issue_count': len(issues)
            }
            
            if issues:
                print(f"   Found {len(issues)} issues:")
                for i, issue in enumerate(issues, 1):
                    print(f"     {i}. {issue}")
                    print(f"        Fix: {fixes[i-1]}")
            else:
                print("   No significant issues found")
        
        # Save issues and fixes
        issues_df = pd.DataFrame([
            {
                'Model': model_name,
                'Issue_Count': data['issue_count'],
                'Issues': '; '.join(data['issues']),
                'Fixes': '; '.join(data['fixes'])
            }
            for model_name, data in issues_and_fixes.items()
        ])
        issues_df.to_csv('/home/ubuntu/issues_and_fixes.csv', index=False)
        
        self.issues_and_fixes = issues_and_fixes
        return issues_and_fixes
    
    def generate_comprehensive_report(self):
        """
        Generate a comprehensive testing and evaluation report.
        """
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE TEST REPORT")
        print("="*60)
        
        # Compile all results
        report_data = {
            'model_scores': self.model_scores,
            'cv_results': getattr(self, 'cv_results', {}),
            'robustness_results': getattr(self, 'robustness_results', {}),
            'clinical_metrics': getattr(self, 'clinical_metrics', {}),
            'error_analysis': getattr(self, 'error_analysis', {}),
            'issues_and_fixes': getattr(self, 'issues_and_fixes', {})
        }
        
        # Create summary statistics
        summary_stats = {}
        
        for model_name in self.models.keys():
            if model_name in self.model_scores.index:
                scores = self.model_scores.loc[model_name]
                
                # Overall performance score (weighted average)
                performance_score = (
                    0.3 * scores['accuracy'] +
                    0.25 * scores['precision'] +
                    0.25 * scores['recall'] +
                    0.2 * scores['auc_roc']
                )
                
                summary_stats[model_name] = {
                    'performance_score': performance_score,
                    'accuracy': scores['accuracy'],
                    'f1_score': scores['f1_score'],
                    'auc_roc': scores['auc_roc'],
                    'issue_count': self.issues_and_fixes.get(model_name, {}).get('issue_count', 0)
                }
        
        # Rank models by performance score
        ranked_models = sorted(summary_stats.items(), key=lambda x: x[1]['performance_score'], reverse=True)
        
        print("Model Performance Ranking:")
        print("="*40)
        for rank, (model_name, stats) in enumerate(ranked_models, 1):
            print(f"{rank:2d}. {model_name:20s} - Score: {stats['performance_score']:.4f}")
            print(f"    Accuracy: {stats['accuracy']:.4f}, F1: {stats['f1_score']:.4f}, AUC: {stats['auc_roc']:.4f}")
            print(f"    Issues: {stats['issue_count']}")
        
        # Save summary
        summary_df = pd.DataFrame(summary_stats).T
        summary_df.to_csv('/home/ubuntu/model_performance_summary.csv')
        
        print(f"\nTesting and evaluation completed!")
        print(f"Best performing model: {ranked_models[0][0]}")
        print(f"Performance score: {ranked_models[0][1]['performance_score']:.4f}")
        
        return report_data, ranked_models

def main():
    """
    Main function to run the complete testing and evaluation pipeline.
    """
    print("AI-BASED DIABETES PREDICTION SYSTEM")
    print("Model Testing and Performance Evaluation Pipeline")
    print("="*60)
    
    # Initialize testing class
    tester = ModelTestingEvaluation()
    
    try:
        # Load data and models
        tester.load_data_and_models()
        
        # Perform cross-validation testing
        cv_results = tester.perform_cross_validation_testing()
        
        # Perform robustness testing
        robustness_results = tester.perform_robustness_testing()
        
        # Perform error analysis
        error_analysis = tester.perform_error_analysis()
        
        # Calculate clinical metrics
        clinical_metrics = tester.calculate_clinical_metrics()
        
        # Generate learning curves
        tester.generate_learning_curves()
        
        # Identify and fix issues
        issues_and_fixes = tester.identify_and_fix_issues()
        
        # Generate comprehensive report
        report_data, ranked_models = tester.generate_comprehensive_report()
        
        print("\n" + "="*60)
        print("TESTING AND EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return tester
        
    except Exception as e:
        print(f"Error in testing and evaluation pipeline: {str(e)}")
        return None

if __name__ == "__main__":
    tester = main()

