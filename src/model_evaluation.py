import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (roc_curve, auc, brier_score_loss,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.calibration import calibration_curve
from scipy import stats
from itertools import combinations
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Parallel processing settings
N_JOBS = -1
BATCH_SIZE = 5000

# =============== Utility Functions: Load Models ===============
def load_models():
    """
    Load saved models (LR, RF, SVM, XGBoost, TabNet).
    Files should be in the saved_models_20000 folder.
    """
    models = {}
    # Load sklearn models
    for name in ['TabNet', 'RF', 'SVM']:
        try:
            models[name] = joblib.load(f'saved_models_20000/{name}_model.joblib')
            print(f"Successfully loaded {name} model")
        except Exception as e:
            print(f"Error loading {name} model: {str(e)}")
    
    # Load XGBoost
    try:
        xgb_model = XGBClassifier()
        xgb_model.load_model('saved_models_20000/XGBoost_model.json')
        models['XGBoost'] = xgb_model
        print("Successfully loaded XGBoost model")
    except Exception as e:
        print(f"Error loading XGBoost model: {str(e)}")
    
    # Load TabNet
    try:
        tabnet_model = TabNetClassifier()
        tabnet_model.load_model('saved_models_20000/LR_model.zip')
        models['LR'] = tabnet_model
        print("Successfully loaded LR model")
    except Exception as e:
        print(f"Error loading LR model: {str(e)}")
    
    if not models:
        raise ValueError("No models were successfully loaded!")
    
    return models

# =============== Utility Functions: Calculate Basic Metrics ===============
def calculate_basic_metrics(y_true, y_pred, y_prob):
    """
    Calculate basic classification metrics
    """
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_prob),
        'Brier': brier_score_loss(y_true, y_prob),
        'Kappa': cohen_kappa_score(y_true, y_pred)  # Add this line
    }

# =============== DeLong Test Related Functions ===============
def delong_roc_test(y_true, y1_pred, y2_pred):
    """
    DeLong test for comparing two ROCs, returns p-value
    """
    unique_classes = np.unique(y_true)
    if len(unique_classes) != 2:
        print(f"Warning: Expected 2 classes, got {len(unique_classes)}")
        return np.nan
    
    # If labels are not [0,1], convert them
    if not np.array_equal(np.sort(unique_classes), np.array([0, 1])):
        y_true = (y_true == unique_classes[1]).astype(int)
    
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        print("Warning: One or both classes have no samples")
        return np.nan
    
    try:
        auc1 = roc_auc_score(y_true, y1_pred)
        auc2 = roc_auc_score(y_true, y2_pred)
        
        var_auc1 = calculate_variance(y1_pred, pos_idx, neg_idx)
        var_auc2 = calculate_variance(y2_pred, pos_idx, neg_idx)
        cov_auc = calculate_covariance(y1_pred, y2_pred, pos_idx, neg_idx)
        
        z = (auc1 - auc2) / np.sqrt(var_auc1 + var_auc2 - 2*cov_auc)
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        return p
    except Exception as e:
        print(f"Error in DeLong test calculation: {str(e)}")
        return np.nan

def calculate_variance(pred, pos_idx, neg_idx):
    pos_scores = pred[pos_idx]
    neg_scores = pred[neg_idx]
    n1, n2 = len(pos_idx), len(neg_idx)
    
    v10 = np.sum([
        (stats.rankdata(np.concatenate([pos_scores, [x]])) <= n1).mean()
        for x in neg_scores
    ]) / n2 - 0.5
    
    v01 = np.sum([
        (stats.rankdata(np.concatenate([neg_scores, [x]])) <= n2).mean()
        for x in pos_scores
    ]) / n1 - 0.5
    
    return (v10 + v01) / (n1 * n2)

def calculate_covariance(pred1, pred2, pos_idx, neg_idx):
    pos_scores1, pos_scores2 = pred1[pos_idx], pred2[pos_idx]
    neg_scores1, neg_scores2 = pred1[neg_idx], pred2[neg_idx]
    n1, n2 = len(pos_idx), len(neg_idx)
    
    cov = (
        np.sum([
            ((stats.rankdata(np.concatenate([pos_scores1, [x]])) <= n1).mean() *
             (stats.rankdata(np.concatenate([pos_scores2, [x]])) <= n1).mean())
            for x in neg_scores1
        ]) / n2 - 0.25
    )
    return cov / (n1 * n2)

# =============== Plotting Functions: ROC, Calibration, DCA ===============
def plot_roc_curves(models_pred_prob, y_true, save_path='figures/roc_curves.png'):
    plt.figure(figsize=(7, 7))
    
    for name, y_prob in models_pred_prob.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], '--', lw=2, color='gray', alpha=.8, label='Reference')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('1 - Specificity', fontsize=16, labelpad=10)
    plt.ylabel('Sensitivity', fontsize=16, labelpad=10)
    plt.title('ROC Curves', fontsize=18, pad=15)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_calibration_curves(models_pred_prob, y_true, save_path='figures/calibration_curves.png'):
    plt.figure(figsize=(7, 7))
    
    for name, y_prob in models_pred_prob.items():
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', markersize=4, label=name)
        
        brier = brier_score_loss(y_true, y_prob)
        print(f"Model {name} Brier Score: {brier:.4f}")
    
    plt.plot([0, 1], [0, 1], '--', lw=2, color='gray', alpha=.8, label='Reference')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('Mean Predicted Probability', fontsize=16, labelpad=10)
    plt.ylabel('Fraction of Positives', fontsize=16, labelpad=10)
    plt.title('Calibration Curves', fontsize=18, pad=15)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def calculate_net_benefit(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    n = len(y_true)
    # If model makes no positive predictions at this threshold, net benefit is 0
    if (tp + fp) == 0:
        return 0
    return (tp/n) - (fp/n) * (threshold / (1 - threshold))

def plot_decision_curves(models_pred_prob, y_true, save_path='figures/decision_curves.png'):
    thresholds = np.arange(0, 1.01, 0.01)
    plt.figure(figsize=(9, 7))
    
    # Treat All
    n = len(y_true)
    n_positive = np.sum(y_true)
    # NB_all = TP_all/n - (FP_all/n)*threshold/(1-threshold)
    #   => When predicting all as positive: TP_all = sum(y_true), FP_all = n - sum(y_true)
    #   => NB_all = (sum(y_true)/n) - [(n - sum(y_true))/n] * ...
    # Simplified formula as follows:
    net_benefit_all = (n_positive / n) - ((n - n_positive) / n) * (thresholds / (1 - thresholds))
    plt.plot(thresholds, net_benefit_all, 'gray', lw=2, ls='--', label='Treat All')
    
    # Treat None: NB=0
    plt.plot(thresholds, np.zeros_like(thresholds), 'gray', lw=2, ls=':', label='Treat None')
    
    # Each model
    for name, y_prob in models_pred_prob.items():
        net_benefits = [calculate_net_benefit(y_true, y_prob, t) for t in thresholds]
        plt.plot(thresholds, net_benefits, label=name, linewidth=2)
    
    plt.xlim([0, 1])
    plt.ylim([-0.02, 0.2])
    plt.xlabel('Threshold Probability', fontsize=16, labelpad=10)
    plt.ylabel('Net Benefit', fontsize=16, labelpad=10)
    plt.title('Decision Curve Analysis', fontsize=18, pad=15)
    plt.grid(True, alpha=0.5)
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

# =============== Parallel Evaluation Functions ===============
def evaluate_model_parallel(model_name, model, X_test, y_test, batch_size=5000):
    """
    Parallel evaluation of single model performance, using batch processing for predictions
    """
    print(f"Evaluating {model_name}...")
    all_predictions = []
    all_probabilities = []
    
    n_samples = len(X_test)
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_X = X_test.iloc[start_idx:end_idx]
        
        # TabNet requires float32 numpy arrays
        if model_name == 'LR':
            X_batch = batch_X.values.astype(np.float32)
            batch_pred = model.predict(X_batch)
            batch_prob = model.predict_proba(X_batch)[:, 1]
        else:
            batch_pred = model.predict(batch_X)
            batch_prob = model.predict_proba(batch_X)[:, 1]
        
        all_predictions.extend(batch_pred)
        all_probabilities.extend(batch_prob)
        
        print(f"{model_name}: Processed {end_idx}/{n_samples} samples")
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate various metrics
    metrics = calculate_basic_metrics(y_test, all_predictions, all_probabilities)
    metrics['predictions'] = all_predictions
    metrics['probabilities'] = all_probabilities
    
    return model_name, metrics

# =============== Confusion Matrix Plotting Function ===============
def plot_confusion_matrices(models, X_test, y_test, save_folder='figures'):
    os.makedirs(save_folder, exist_ok=True)
    
    for model_name, model in models.items():
        print(f"Plotting confusion matrix for: {model_name}")
        
        if model_name == 'LR':
            X_input = X_test.values.astype(np.float32)
        else:
            X_input = X_test
        
        y_pred = model.predict(X_input)
        
        cm = confusion_matrix(y_test, y_pred, labels=[0,1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap='Blues', ax=ax, values_format='d')
        
        plt.title(f'{model_name} Confusion Matrix', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14, labelpad=10)
        plt.ylabel('True Label', fontsize=14, labelpad=10)
        
        for text in ax.texts:
            text.set_fontsize(18)
        
        save_path = os.path.join(save_folder, f'{model_name}_Confusion_Matrix.jpg')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f'{model_name} Confusion Matrix:\n{cm}\n')

# =============== Feature Importance Extraction and Plotting ===============
def get_feature_importances(model, model_name):
    """
    Extract feature importance based on model type; return None if unavailable.
    """
    # Corresponding to our loading names: 'logistic_regression','random_forest','svm','xgboost','tabnet'
    if model_name == 'TabNet':
        # LogisticRegression.coef_: shape (1, n_features)
        # Use absolute values as importance
        return np.abs(model.coef_[0])
    
    elif model_name == 'RF':
        return model.feature_importances_
    
    elif model_name == 'SVM':
        # SVC(probability=True) does not provide feature_importances_
        return None
    
    elif model_name == 'XGBoost':
        # XGBClassifier has feature_importances_
        return model.feature_importances_
    
    elif model_name == 'LR':
        # TabNetClassifier has feature_importances_ after fit
        return model.feature_importances_
    
    return None

def plot_top_features(importances, feature_names, model_name, top_n=10, save_path='figures'):
    """
    Plot bar chart of top_n feature importances
    """
    df_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    df_imp = df_imp.sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_imp, x='importance', y='feature', palette='Blues_r')
    plt.title(f'Feature Importances (Top {top_n}) - {model_name}', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    filename = f'{model_name}_top{top_n}_features.png'
    plt.savefig(os.path.join(save_path, filename), dpi=300)
    plt.show()

# =========== Ablation Study Functions ===========
def perform_ablation_study(best_model, best_model_name, X_test, y_test, feature_names):
    """Perform ablation study: set each feature to 0 while keeping others at original values"""
    # Get feature importance
    importances = get_feature_importances(best_model, best_model_name)
    if importances is None:
        print(f"Cannot perform ablation study for {best_model_name}: model does not provide feature importance")
        return None
        
    # Get top 5 most important features
    top_features_idx = np.argsort(importances)[-5:][::-1]
    top_features = feature_names[top_features_idx]
    
    print(f"\nTop 5 most important features: {', '.join(top_features)}")
    
    # Prepare to record results
    ablation_results = []
    
    # Baseline performance (using all features)
    X_current = X_test.copy()
    
    if best_model_name == 'LR':
        X_input = X_current.values.astype(np.float32)
    else:
        X_input = X_current
        
    y_pred = best_model.predict(X_input)
    y_prob = best_model.predict_proba(X_input)[:, 1]
    
    metrics = {
        'Removed_Feature': 'None',
        'Remaining_Features': len(feature_names),
        **calculate_basic_metrics(y_test, y_pred, y_prob)
    }
    ablation_results.append(metrics)
    
    # Sequentially ablate individual important features
    for feature in top_features:
        print(f"\nAblating feature: {feature}")
        
        # Re-copy original data each time, only set current feature to 0
        X_current = X_test.copy()
        X_current[feature] = 0
        
        if best_model_name == 'LR':
            X_input = X_current.values.astype(np.float32)
        else:
            X_input = X_current
            
        y_pred = best_model.predict(X_input)
        y_prob = best_model.predict_proba(X_input)[:, 1]
        
        metrics = {
            'Removed_Feature': feature,
            'Remaining_Features': len(feature_names) - 1,  # Only subtract the current ablated feature
            **calculate_basic_metrics(y_test, y_pred, y_prob)
        }
        ablation_results.append(metrics)
    
    # Convert to DataFrame and save
    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_excel('results/ablation_study.xlsx', index=False)
    
    # Print results
    print("\n====== Ablation Study Results ======")
    print("Model performance after individually ablating each important feature:")
    selected_metrics = ['Removed_Feature', 'Accuracy', 'Precision', 'Recall', 'F1', 'Kappa']
    print(ablation_df[selected_metrics].to_string(index=False))
    
    return ablation_df

# =============== Main Function ===============
def main():
    # Create folders to save results
    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load test data
    print("Loading test data...")
    test_data = pd.read_excel("StandardizedTestingData3.xlsx")
    X_test = test_data.drop('Group', axis=1)
    y_test = test_data['Group']
    
    print(f"Dataset size: {len(X_test)} samples")
    
    # Load models
    print("Loading models...")
    models = load_models()
    
    # Parallel evaluation of all models
    print("Starting parallel evaluation...")
    results = Parallel(n_jobs=N_JOBS)(
        delayed(evaluate_model_parallel)(name, model, X_test, y_test, BATCH_SIZE)
        for name, model in models.items()
    )
    
    # Collect results
    models_pred = {}       # model_name -> predictions
    models_pred_prob = {}  # model_name -> probabilities
    basic_metrics = {}     # model_name -> dict of metrics
    
    for name, metrics in results:
        models_pred[name] = metrics.pop('predictions')
        models_pred_prob[name] = metrics.pop('probabilities')
        basic_metrics[name] = metrics
    
    # DeLong test
    print("Performing DeLong tests...")
    model_names = list(models.keys())
    delong_results = []
    
    # Batch DeLong tests (optional), consistent with batch prediction
    for batch_start in range(0, len(y_test), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(y_test))
        batch_y = y_test[batch_start:batch_end]
        
        batch_delong = Parallel(n_jobs=N_JOBS)(
            delayed(delong_roc_test)(
                batch_y,
                models_pred_prob[m1][batch_start:batch_end],
                models_pred_prob[m2][batch_start:batch_end]
            )
            for m1, m2 in combinations(model_names, 2)
        )
        delong_results.append(batch_delong)
    
    # Aggregate DeLong results (average p-values)
    final_delong = []
    combo_list = list(combinations(model_names, 2))
    for i, (m1, m2) in enumerate(combo_list):
        pvals = []
        for batch in delong_results:
            if not np.isnan(batch[i]):
                pvals.append(batch[i])
        p_mean = np.mean(pvals) if len(pvals) > 0 else np.nan
        
        final_delong.append({'Model 1': m1, 'Model 2': m2, 'p-value': p_mean})
    
    # Generate plots
    print("Generating ROC, calibration, and decision curves...")
    plot_roc_curves(models_pred_prob, y_test, save_path='figures/roc_curves.png')
    plot_calibration_curves(models_pred_prob, y_test, save_path='figures/calibration_curves.png')
    plot_decision_curves(models_pred_prob, y_test, save_path='figures/decision_curves.png')
    
    # Confusion matrices
    print("Generating confusion matrices...")
    plot_confusion_matrices(models, X_test, y_test, save_folder='figures')
    
    # Save results
    print("Saving results...")
    pd.DataFrame(basic_metrics).T.to_excel('results/basic_metrics.xlsx')
    pd.DataFrame(final_delong).to_excel('results/delong_test_results.xlsx', index=False)
    
    print("\nEvaluation complete! Results have been saved.")
    
    # Print tables
    metrics_df = pd.DataFrame(basic_metrics).T
    print("\nBasic metrics:")
    print(metrics_df)
    print("\nDeLong test results:")
    print(pd.DataFrame(final_delong))
    
    # =========== New addition: Automatically select highest AUC model, display top 10 feature importance ===========
    print("\nSelecting best model by AUC and plotting top-10 feature importances...")
    best_model_name = metrics_df['AUC'].idxmax()
    best_model_auc = metrics_df.loc[best_model_name, 'AUC']
    print(f"Best model: {best_model_name} (AUC = {best_model_auc:.4f})")
    
    best_model = models[best_model_name]
    importances = get_feature_importances(best_model, best_model_name)
    
    if importances is None:
        print(f"Model '{best_model_name}' does not provide feature importances.")
    else:
        # Plot top 10 feature importances
        plot_top_features(importances, X_test.columns, best_model_name, top_n=10, save_path='figures')
    
    print("\nAll tasks completed!")

    # Add at the end of main function:
    print("\nPerforming ablation study...")
    best_model = models[best_model_name]
    ablation_results = perform_ablation_study(
        best_model, 
        best_model_name,
        X_test, 
        y_test,
        np.array(X_test.columns)
    )
    
    
if __name__ == "__main__":
    main()
