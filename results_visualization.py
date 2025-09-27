#!/usr/bin/env python3
"""
AI-Based Diabetes Prediction System - Results Visualization and Analysis
========================================================================

This module creates comprehensive visualizations of model performance, analysis results,
and generates publication-ready charts and graphs for the project report.

Author: AI-Based Diabetes Prediction System Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

class ResultsVisualizer:
    """
    Comprehensive results visualization class.
    """
    
    def __init__(self):
        """
        Initialize the visualizer.
        """
        self.models = {}
        self.X_test = None
        self.y_test = None
        self.model_scores = None
        self.feature_names = None
        
    def load_data_and_results(self):
        """
        Load all data and results for visualization.
        """
        print("Loading data and results for visualization...")
        
        # Load test data
        self.X_test = pd.read_csv('/home/ubuntu/X_test.csv')
        self.y_test = pd.read_csv('/home/ubuntu/y_test.csv').squeeze()
        
        # Load model scores
        self.model_scores = pd.read_csv('/home/ubuntu/model_comparison.csv')
        
        # Load feature names
        with open('/home/ubuntu/feature_names.txt', 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
        
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
                    except Exception as e:
                        print(f"   Error loading {model_name}: {str(e)}")
        
        print(f"Loaded {len(self.models)} models and test data with {len(self.X_test)} samples")
        return True
    
    def create_model_performance_comparison(self):
        """
        Create comprehensive model performance comparison charts.
        """
        print("\nCreating model performance comparison charts...")
        
        # Create a large figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Sort models by F1-Score for consistent ordering
        sorted_scores = self.model_scores.sort_values('F1-Score', ascending=True)
        
        # 1. Accuracy Comparison
        axes[0, 0].barh(sorted_scores['Model'], sorted_scores['Accuracy'], color='skyblue')
        axes[0, 0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Accuracy')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # 2. Precision vs Recall
        axes[0, 1].scatter(sorted_scores['Recall'], sorted_scores['Precision'], 
                          s=100, alpha=0.7, c=range(len(sorted_scores)), cmap='viridis')
        for i, model in enumerate(sorted_scores['Model']):
            axes[0, 1].annotate(model, 
                               (sorted_scores.iloc[i]['Recall'], sorted_scores.iloc[i]['Precision']),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[0, 1].set_title('Precision vs Recall', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. F1-Score Comparison
        axes[0, 2].barh(sorted_scores['Model'], sorted_scores['F1-Score'], color='lightcoral')
        axes[0, 2].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('F1-Score')
        axes[0, 2].grid(axis='x', alpha=0.3)
        
        # 4. AUC-ROC Comparison
        axes[1, 0].barh(sorted_scores['Model'], sorted_scores['AUC-ROC'], color='lightgreen')
        axes[1, 0].set_title('AUC-ROC Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('AUC-ROC')
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # 5. Specificity Comparison
        axes[1, 1].barh(sorted_scores['Model'], sorted_scores['Specificity'], color='orange')
        axes[1, 1].set_title('Specificity Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Specificity')
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        # 6. Overall Performance Heatmap (for top 5 models)
        top_5_models = sorted_scores.tail(5)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Specificity']
        
        # Create performance heatmap
        heatmap_data = top_5_models[metrics].T
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=top_5_models['Model'], yticklabels=metrics,
                   ax=axes[1, 2], cbar_kws={'shrink': 0.8})
        
        axes[1, 2].set_title('Top 5 Models - Performance Heatmap', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Models')
        axes[1, 2].set_ylabel('Metrics')
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Model performance comparison saved as 'model_performance_comparison.png'")
    
    def create_confusion_matrices(self):
        """
        Create confusion matrices for top performing models.
        """
        print("\nCreating confusion matrices...")
        
        # Select top 6 models based on F1-Score
        top_models = self.model_scores.nlargest(6, 'F1-Score')['Model'].tolist()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, model_name in enumerate(top_models):
            if model_name in self.models:
                try:
                    # Make predictions
                    model = self.models[model_name]
                    y_pred = model.predict(self.X_test)
                    
                    # Create confusion matrix
                    cm = confusion_matrix(self.y_test, y_pred)
                    
                    # Plot confusion matrix
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=['Non-Diabetic', 'Diabetic'],
                               yticklabels=['Non-Diabetic', 'Diabetic'],
                               ax=axes[i])
                    
                    axes[i].set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
                    axes[i].set_xlabel('Predicted')
                    axes[i].set_ylabel('Actual')
                    
                    # Add performance metrics as text
                    tn, fp, fn, tp = cm.ravel()
                    accuracy = (tp + tn) / (tp + tn + fp + fn)
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    axes[i].text(0.02, 0.98, f'Acc: {accuracy:.3f}\nPrec: {precision:.3f}\nRec: {recall:.3f}\nF1: {f1:.3f}',
                               transform=axes[i].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                except Exception as e:
                    axes[i].text(0.5, 0.5, f'Error: {str(e)}', 
                               transform=axes[i].transAxes, ha='center', va='center')
                    axes[i].set_title(f'{model_name} - Error')
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Confusion matrices saved as 'confusion_matrices.png'")
    
    def create_roc_curves(self):
        """
        Create ROC curves for all models.
        """
        print("\nCreating ROC curves...")
        
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.models)))
        
        for i, (model_name, model) in enumerate(self.models.items()):
            try:
                # Get prediction probabilities
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_pred_proba = model.decision_function(self.X_test)
                else:
                    continue
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
                auc_score = np.trapz(tpr, fpr)
                
                # Plot ROC curve
                plt.plot(fpr, tpr, color=colors[i], linewidth=2,
                        label=f'{model_name} (AUC = {auc_score:.3f})')
                
            except Exception as e:
                print(f"   Error creating ROC curve for {model_name}: {str(e)}")
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("ROC curves saved as 'roc_curves.png'")
    
    def create_precision_recall_curves(self):
        """
        Create Precision-Recall curves for all models.
        """
        print("\nCreating Precision-Recall curves...")
        
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.models)))
        
        for i, (model_name, model) in enumerate(self.models.items()):
            try:
                # Get prediction probabilities
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_pred_proba = model.decision_function(self.X_test)
                else:
                    continue
                
                # Calculate Precision-Recall curve
                precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
                ap_score = np.trapz(precision, recall)
                
                # Plot PR curve
                plt.plot(recall, precision, color=colors[i], linewidth=2,
                        label=f'{model_name} (AP = {ap_score:.3f})')
                
            except Exception as e:
                print(f"   Error creating PR curve for {model_name}: {str(e)}")
        
        # Plot baseline
        baseline = self.y_test.mean()
        plt.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, 
                   label=f'Baseline (AP = {baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - All Models', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Precision-Recall curves saved as 'precision_recall_curves.png'")
    
    def create_feature_importance_analysis(self):
        """
        Create feature importance analysis for tree-based models.
        """
        print("\nCreating feature importance analysis...")
        
        # Select models that have feature importance
        tree_models = ['Random_Forest', 'XGBoost', 'LightGBM', 'Decision_Tree']
        available_models = [model for model in tree_models if model in self.models]
        
        if not available_models:
            print("No tree-based models available for feature importance analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.ravel()
        
        for i, model_name in enumerate(available_models[:4]):
            try:
                model = self.models[model_name]
                
                # Get feature importance
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importance = np.abs(model.coef_[0])
                else:
                    continue
                
                # Create feature importance DataFrame
                feature_importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=True)
                
                # Plot top 15 features
                top_features = feature_importance_df.tail(15)
                
                axes[i].barh(top_features['feature'], top_features['importance'], color='steelblue')
                axes[i].set_title(f'{model_name}\nTop 15 Feature Importance', fontweight='bold')
                axes[i].set_xlabel('Importance')
                axes[i].grid(axis='x', alpha=0.3)
                
                # Rotate y-axis labels for better readability
                axes[i].tick_params(axis='y', labelsize=10)
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error: {str(e)}', 
                           transform=axes[i].transAxes, ha='center', va='center')
                axes[i].set_title(f'{model_name} - Error')
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Feature importance analysis saved as 'feature_importance.png'")
    
    def create_model_comparison_summary(self):
        """
        Create a comprehensive model comparison summary visualization.
        """
        print("\nCreating model comparison summary...")
        
        # Load additional results
        try:
            cv_results = pd.read_csv('/home/ubuntu/cross_validation_results.csv', index_col=0)
            clinical_metrics = pd.read_csv('/home/ubuntu/clinical_metrics.csv', index_col=0)
            performance_summary = pd.read_csv('/home/ubuntu/model_performance_summary.csv', index_col=0)
        except:
            print("Some result files not found, creating basic summary")
            cv_results = None
            clinical_metrics = None
            performance_summary = None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Performance Score Ranking
        if performance_summary is not None:
            sorted_performance = performance_summary.sort_values('performance_score', ascending=True)
            axes[0, 0].barh(sorted_performance.index, sorted_performance['performance_score'], 
                           color='lightblue')
            axes[0, 0].set_title('Overall Performance Score Ranking', fontweight='bold')
            axes[0, 0].set_xlabel('Performance Score')
            axes[0, 0].grid(axis='x', alpha=0.3)
        
        # 2. Cross-Validation Stability
        if cv_results is not None:
            cv_f1_mean = cv_results['f1_mean'].sort_values(ascending=True)
            cv_f1_std = cv_results.loc[cv_f1_mean.index, 'f1_std']
            
            axes[0, 1].barh(cv_f1_mean.index, cv_f1_mean.values, 
                           xerr=cv_f1_std.values, color='lightgreen', alpha=0.7)
            axes[0, 1].set_title('Cross-Validation F1-Score (Mean ± Std)', fontweight='bold')
            axes[0, 1].set_xlabel('F1-Score')
            axes[0, 1].grid(axis='x', alpha=0.3)
        
        # 3. Clinical Metrics Comparison
        if clinical_metrics is not None:
            # Select key clinical metrics
            key_metrics = ['sensitivity', 'specificity', 'ppv', 'npv']
            available_metrics = [m for m in key_metrics if m in clinical_metrics.columns]
            
            if available_metrics:
                clinical_subset = clinical_metrics[available_metrics].fillna(0)
                
                # Create heatmap
                sns.heatmap(clinical_subset, annot=True, fmt='.3f', cmap='YlOrRd',
                           ax=axes[1, 0], cbar_kws={'shrink': 0.8})
                axes[1, 0].set_title('Clinical Metrics Heatmap', fontweight='bold')
                axes[1, 0].set_xlabel('Metrics')
                axes[1, 0].set_ylabel('Models')
        
        # 4. Model Complexity vs Performance
        model_complexity = {
            'Logistic_Regression': 1,
            'Naive_Bayes': 1,
            'KNN': 2,
            'Decision_Tree': 3,
            'SVM': 4,
            'Neural_Network': 5,
            'Random_Forest': 6,
            'XGBoost': 7,
            'LightGBM': 7,
            'Voting_Soft': 8,
            'Voting_Hard': 8,
            'Stacking': 9
        }
        
        # Get F1-scores for complexity analysis
        f1_scores = {}
        for _, row in self.model_scores.iterrows():
            f1_scores[row['Model']] = row['F1-Score']
        
        complexity_data = []
        performance_data = []
        model_names = []
        
        for model_name in f1_scores.keys():
            if model_name in model_complexity:
                complexity_data.append(model_complexity[model_name])
                performance_data.append(f1_scores[model_name])
                model_names.append(model_name)
        
        scatter = axes[1, 1].scatter(complexity_data, performance_data, 
                                   s=100, alpha=0.7, c=range(len(model_names)), cmap='viridis')
        
        for i, name in enumerate(model_names):
            axes[1, 1].annotate(name, (complexity_data[i], performance_data[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        axes[1, 1].set_title('Model Complexity vs Performance', fontweight='bold')
        axes[1, 1].set_xlabel('Model Complexity (1=Simple, 9=Complex)')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/model_comparison_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Model comparison summary saved as 'model_comparison_summary.png'")
    
    def create_data_insights_visualization(self):
        """
        Create visualizations showing data insights and patterns.
        """
        print("\nCreating data insights visualization...")
        
        # Load processed data
        processed_data = pd.read_csv('/home/ubuntu/processed_diabetes_data.csv')
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Target distribution
        target_counts = processed_data['Outcome'].value_counts()
        axes[0, 0].pie(target_counts.values, labels=['Non-Diabetic', 'Diabetic'], 
                      autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
        axes[0, 0].set_title('Target Variable Distribution', fontweight='bold')
        
        # 2. Age distribution by outcome
        diabetic = processed_data[processed_data['Outcome'] == 1]['Age']
        non_diabetic = processed_data[processed_data['Outcome'] == 0]['Age']
        
        axes[0, 1].hist(non_diabetic, bins=20, alpha=0.7, label='Non-Diabetic', color='lightblue')
        axes[0, 1].hist(diabetic, bins=20, alpha=0.7, label='Diabetic', color='lightcoral')
        axes[0, 1].set_title('Age Distribution by Outcome', fontweight='bold')
        axes[0, 1].set_xlabel('Age')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # 3. Glucose vs BMI scatter plot
        scatter = axes[0, 2].scatter(processed_data['Glucose'], processed_data['BMI'], 
                                   c=processed_data['Outcome'], cmap='coolwarm', alpha=0.6)
        axes[0, 2].set_title('Glucose vs BMI by Outcome', fontweight='bold')
        axes[0, 2].set_xlabel('Glucose Level')
        axes[0, 2].set_ylabel('BMI')
        plt.colorbar(scatter, ax=axes[0, 2], label='Outcome')
        
        # 4. Correlation heatmap of key features
        key_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'Age', 'Outcome']
        correlation_matrix = processed_data[key_features].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', ax=axes[1, 0])
        axes[1, 0].set_title('Feature Correlation Matrix', fontweight='bold')
        
        # 5. Risk score distribution
        if 'Risk_Score' in processed_data.columns:
            diabetic_risk = processed_data[processed_data['Outcome'] == 1]['Risk_Score']
            non_diabetic_risk = processed_data[processed_data['Outcome'] == 0]['Risk_Score']
            
            axes[1, 1].hist(non_diabetic_risk, bins=20, alpha=0.7, label='Non-Diabetic', color='lightblue')
            axes[1, 1].hist(diabetic_risk, bins=20, alpha=0.7, label='Diabetic', color='lightcoral')
            axes[1, 1].set_title('Risk Score Distribution', fontweight='bold')
            axes[1, 1].set_xlabel('Risk Score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
        
        # 6. Feature distribution comparison
        feature_means_diabetic = processed_data[processed_data['Outcome'] == 1][key_features[:-1]].mean()
        feature_means_non_diabetic = processed_data[processed_data['Outcome'] == 0][key_features[:-1]].mean()
        
        x = np.arange(len(key_features[:-1]))
        width = 0.35
        
        axes[1, 2].bar(x - width/2, feature_means_non_diabetic, width, 
                      label='Non-Diabetic', color='lightblue', alpha=0.7)
        axes[1, 2].bar(x + width/2, feature_means_diabetic, width, 
                      label='Diabetic', color='lightcoral', alpha=0.7)
        
        axes[1, 2].set_title('Mean Feature Values by Outcome', fontweight='bold')
        axes[1, 2].set_xlabel('Features')
        axes[1, 2].set_ylabel('Mean Value (Normalized)')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(key_features[:-1], rotation=45)
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/data_insights.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Data insights visualization saved as 'data_insights.png'")

def main():
    """
    Main function to run the complete visualization pipeline.
    """
    print("AI-BASED DIABETES PREDICTION SYSTEM")
    print("Results Visualization and Analysis Pipeline")
    print("="*60)
    
    # Initialize visualizer
    visualizer = ResultsVisualizer()
    
    try:
        # Load data and results
        visualizer.load_data_and_results()
        
        # Create all visualizations
        visualizer.create_model_performance_comparison()
        visualizer.create_confusion_matrices()
        visualizer.create_roc_curves()
        visualizer.create_precision_recall_curves()
        visualizer.create_feature_importance_analysis()
        visualizer.create_model_comparison_summary()
        visualizer.create_data_insights_visualization()
        
        print("\n" + "="*60)
        print("VISUALIZATION PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Generated visualizations:")
        print("  - model_performance_comparison.png")
        print("  - confusion_matrices.png")
        print("  - roc_curves.png")
        print("  - precision_recall_curves.png")
        print("  - feature_importance.png")
        print("  - model_comparison_summary.png")
        print("  - data_insights.png")
        
        return visualizer
        
    except Exception as e:
        print(f"Error in visualization pipeline: {str(e)}")
        return None

if __name__ == "__main__":
    visualizer = main()

