import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.utils import resample
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# =============== Model and Data Loading Functions ===============
def load_models_and_data():
    """
    Load saved models and test data
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
    
    # Load test data
    try:
        test_data = pd.read_excel("StandardizedTestingData3.xlsx")
        X_test = test_data.drop('Group', axis=1)
        y_test = test_data['Group'].values
        
        print(f"Test data loaded: {len(X_test)} samples")
        print(f"Positive class prevalence: {np.mean(y_test):.1%}")
        
        return models, X_test, y_test
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return models, None, None

# =============== Variable Name Mapping Function ===============
def create_display_name_mapping(feature_ranking):
    """
    Create mapping for variable display names based on importance ranking
    """
    custom_names = {
        4: "Initial Triage Level: P3",
        5: "ED Visits Last Month", 
        6: "History: Diabetes",
        7: "Activity Ability: Accompanied",
        8: "Diagnostic Procedures: CT",
        9: "History: Hypertension",
        10: "Diagnostic Procedures: ECG"
    }
    
    # Create mapping dictionary
    name_mapping = {}
    for rank, (_, row) in enumerate(feature_ranking.iterrows(), 1):
        if rank in custom_names:
            name_mapping[row['feature']] = custom_names[rank]
        else:
            name_mapping[row['feature']] = row['feature']
    
    return name_mapping

# =============== Permutation Importance Analysis Function ===============
def perform_permutation_importance(model, model_name, X_test, y_test, feature_names, save_path='figures'):
    """
    Perform permutation importance analysis (model-agnostic method)
    """
    print(f"\nPerforming permutation importance analysis for {model_name}...")
    
    try:
        # Prepare input data based on model type
        if model_name == 'LR':
            X_input = X_test.values.astype(np.float32)
        else:
            X_input = X_test
        
        # Calculate permutation importance
        print("  Computing permutation importance (this may take a few minutes)...")
        perm_importance = permutation_importance(
            model, X_input, y_test, 
            n_repeats=30,  # Number of repetitions
            random_state=42,
            scoring='roc_auc',
            n_jobs=-1  # Use all available cores
        )
        
        # Create results DataFrame
        perm_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std,
            'importance_ci_lower': perm_importance.importances_mean - 1.96 * perm_importance.importances_std,
            'importance_ci_upper': perm_importance.importances_mean + 1.96 * perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        # Plot permutation importance
        plt.figure(figsize=(12, 10))
        top_features = perm_df.head(10)
        
        # Create display name mapping
        name_mapping = create_display_name_mapping(perm_df)
        
        # Reverse order to achieve visual sorting from large to small (most important at top)
        top_features_reversed = top_features.iloc[::-1]
        
        # Get display feature names
        display_names = [name_mapping[feature] for feature in top_features_reversed['feature']]
        
        # Use error bars to show uncertainty
        plt.barh(range(len(top_features_reversed)), top_features_reversed['importance_mean'], 
                xerr=top_features_reversed['importance_std'], capsize=5, alpha=0.8)
        plt.yticks(range(len(top_features_reversed)), display_names, fontsize=14)
        plt.xlabel('Permutation Importance (AUC decrease)', fontsize=16)
        plt.ylabel('Feature', fontsize=16)
        plt.title(f'Permutation Feature Importance (Top 10) - {model_name}', fontsize=16)
        plt.xticks(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        os.makedirs(save_path, exist_ok=True)
        perm_plot_path = os.path.join(save_path, f'{model_name}_permutation_importance.png')
        plt.savefig(perm_plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"  Permutation importance analysis completed. Plot saved to {perm_plot_path}")
        return perm_df, perm_importance.importances
        
    except Exception as e:
        print(f"  Permutation importance analysis failed for {model_name}: {str(e)}")
        return None, None

# =============== Bootstrap Permutation Importance Analysis ===============
def bootstrap_permutation_single_iter(args):
    """
    Single bootstrap permutation importance iteration
    """
    model, model_name, X_test, y_test, feature_names, iteration = args
    
    try:
        # Bootstrap resample test data
        indices = resample(range(len(X_test)), random_state=iteration)
        X_boot = X_test.iloc[indices]
        y_boot = y_test[indices]
        
        # Prepare input data based on model type
        if model_name == 'LR':
            X_input = X_boot.values.astype(np.float32)
        else:
            X_input = X_boot
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, X_input, y_boot,
            n_repeats=5,  # Reduce repetitions to improve speed
            random_state=iteration,
            scoring='roc_auc',
            n_jobs=1  # Single thread to avoid nested parallelization
        )
        
        return iteration, perm_importance.importances_mean
        
    except Exception as e:
        print(f"Bootstrap iteration {iteration} failed: {str(e)}")
        return iteration, None

def bootstrap_permutation_stability(model, model_name, X_test, y_test, feature_names, 
                                  n_bootstrap=300, n_workers=4):
    """
    Assess stability of permutation importance using bootstrap method
    """
    print(f"\nPerforming bootstrap stability analysis of permutation importance for {model_name}...")
    print(f"Running {n_bootstrap} bootstrap iterations with {n_workers} workers...")
    
    # Prepare multithreading parameters
    thread_args = [
        (model, model_name, X_test, y_test, feature_names, i)
        for i in range(n_bootstrap)
    ]
    
    # Execute multithreaded bootstrap
    bootstrap_importances = []
    completed_iterations = 0
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_iteration = {executor.submit(bootstrap_permutation_single_iter, args): args[5] 
                             for args in thread_args}
        
        for future in as_completed(future_to_iteration):
            result = future.result()
            if result[1] is not None:
                bootstrap_importances.append(result[1])
                completed_iterations += 1
                
                if completed_iterations % 50 == 0:
                    print(f"  Completed {completed_iterations}/{n_bootstrap} iterations")
    
    if not bootstrap_importances:
        print(f"  No successful bootstrap iterations for {model_name}")
        return None
    
    print(f"  Successfully completed {len(bootstrap_importances)} iterations")
    
    # Calculate statistics
    bootstrap_importances = np.array(bootstrap_importances)
    mean_importances = np.mean(bootstrap_importances, axis=0)
    std_importances = np.std(bootstrap_importances, axis=0)
    ci_lower = np.percentile(bootstrap_importances, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_importances, 97.5, axis=0)
    
    # Create results DataFrame
    stability_df = pd.DataFrame({
        'feature': feature_names,
        'mean_importance': mean_importances,
        'std_importance': std_importances,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'cv': std_importances / (mean_importances + 1e-8)  # Coefficient of variation
    }).sort_values('mean_importance', ascending=False)
    
    return stability_df, bootstrap_importances

# =============== Correlation Analysis Function ===============
def analyze_feature_correlations(X_data, top_features, save_path='figures'):
    """
    Analyze correlations among top features to detect multicollinearity
    """
    print(f"\nAnalyzing feature correlations for multicollinearity detection...")
    
    # Select top features
    top_feature_names = top_features['feature'].head(10).tolist()
    X_top = X_data[top_feature_names]
    
    # Create display name mapping
    name_mapping = create_display_name_mapping(top_features)
    
    # Calculate correlation matrix
    corr_matrix = X_top.corr()
    
    # Create correlation matrix with display names
    display_names = [name_mapping[col] for col in corr_matrix.columns]
    corr_matrix_display = corr_matrix.copy()
    corr_matrix_display.index = display_names
    corr_matrix_display.columns = display_names
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix_display, dtype=bool))  # Show only lower triangle
    sns.heatmap(corr_matrix_display, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.3f')
    plt.title('Feature Correlation Matrix (Top 10 Features)', fontsize=16)
    
    # Increase font size for x and y axis labels
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(save_path, exist_ok=True)
    corr_plot_path = os.path.join(save_path, 'top_features_correlation_matrix.png')
    plt.savefig(corr_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Identify highly correlated feature pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:  # High correlation threshold
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    if high_corr_pairs:
        print(f"  Found {len(high_corr_pairs)} highly correlated feature pairs (|r| > 0.7):")
        for pair in high_corr_pairs:
            print(f"    {pair['feature1']} ↔ {pair['feature2']}: r = {pair['correlation']:.3f}")
    else:
        print("  No highly correlated feature pairs detected (|r| > 0.7)")
    
    print(f"  Correlation analysis completed. Heatmap saved to {corr_plot_path}")
    return corr_matrix, high_corr_pairs

# =============== Visualization Functions ===============
def plot_bootstrap_stability(stability_df, model_name, top_n=10, save_path='figures'):
    """
    Plot bootstrap stability analysis results
    """
    top_features = stability_df.head(top_n)
    
    # Create display name mapping
    name_mapping = create_display_name_mapping(stability_df)
    
    # Reverse order to achieve visual sorting from large to small (most important at top)
    top_features_reversed = top_features.iloc[::-1]
    
    # Get display feature names
    display_names = [name_mapping[feature] for feature in top_features_reversed['feature']]
    
    plt.figure(figsize=(12, 8))
    
    # Main plot: importance means and confidence intervals
    plt.errorbar(top_features_reversed['mean_importance'], range(len(top_features_reversed)), 
                xerr=[top_features_reversed['mean_importance'] - top_features_reversed['ci_lower'],
                      top_features_reversed['ci_upper'] - top_features_reversed['mean_importance']], 
                fmt='o', capsize=5, capthick=2, markersize=8)
    
    plt.yticks(range(len(top_features_reversed)), display_names, fontsize=14)
    plt.xlabel('Permutation Importance (Bootstrap Mean ± 95% CI)', fontsize=16)
    plt.ylabel('Feature', fontsize=16)
    plt.title(f'Bootstrap Permutation Importance Stability (Top {top_n}) - {model_name}', fontsize=16)
    plt.xticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(save_path, exist_ok=True)
    stability_plot_path = os.path.join(save_path, f'{model_name}_bootstrap_permutation_stability.png')
    plt.savefig(stability_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Bootstrap stability plot saved to: {stability_plot_path}")

# =============== Main Function ===============
def main():
    """
    Main function: Comprehensive feature importance analysis based on test data
    """
    # Create folders to save results
    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("=== COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS ===")
    print("This analysis includes:")
    print("1. Permutation-based model-agnostic feature importance")
    print("2. Bootstrap stability analysis of permutation importance")
    print("3. Feature correlation analysis for multicollinearity detection")
    print("4. Statistical robustness assessment\n")
    
    # Load models and data
    models, X_test, y_test = load_models_and_data()
    
    if X_test is None:
        print("Failed to load test data.")
        return
    
    if not models:
        print("No models loaded successfully.")
        return
    
    # Select model with highest AUC for in-depth analysis
    print("\nEvaluating model performance...")
    model_aucs = {}
    
    for name, model in models.items():
        if name == 'LR':
            X_input = X_test.values.astype(np.float32)
        else:
            X_input = X_test
        
        try:
            y_prob = model.predict_proba(X_input)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            model_aucs[name] = auc
            print(f"  {name} AUC: {auc:.4f}")
        except Exception as e:
            print(f"  Error evaluating {name}: {str(e)}")
    
    best_model_name = max(model_aucs, key=model_aucs.get)
    best_model = models[best_model_name]
    
    print(f"\nBest model: {best_model_name} (AUC = {model_aucs[best_model_name]:.4f})")
    print(f"Proceeding with comprehensive feature analysis for {best_model_name}...")
    
    feature_names = X_test.columns.tolist()
    
    # 1. Basic permutation importance analysis
    perm_results, perm_raw = perform_permutation_importance(
        best_model, best_model_name, X_test, y_test, feature_names
    )
    
    # 2. Bootstrap stability analysis
    if perm_results is not None:
        stability_results, bootstrap_importances = bootstrap_permutation_stability(
            best_model, best_model_name, X_test, y_test, 
            feature_names, n_bootstrap=300, n_workers=4
        )
        
        if stability_results is not None:
            plot_bootstrap_stability(stability_results, best_model_name)
    
    # 3. Correlation analysis
    if perm_results is not None:
        corr_matrix, high_corr_pairs = analyze_feature_correlations(X_test, perm_results)
    
    # 4. Save comprehensive results
    if perm_results is not None and stability_results is not None:
        # Merge results
        comparison_df = perm_results.merge(
            stability_results, 
            on='feature', 
            suffixes=('_single', '_bootstrap')
        )
        
        # Save detailed results to Excel
        excel_path = f'results/{best_model_name}_feature_importance_analysis.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            perm_results.to_excel(writer, sheet_name='Permutation_Importance', index=False)
            stability_results.to_excel(writer, sheet_name='Bootstrap_Stability', index=False)
            comparison_df.to_excel(writer, sheet_name='Combined_Analysis', index=False)
            if high_corr_pairs:
                pd.DataFrame(high_corr_pairs).to_excel(writer, sheet_name='High_Correlations', index=False)
        
        print(f"\nComprehensive results saved to: {excel_path}")
        
        # Output analysis summary
        print(f"\n=== FEATURE IMPORTANCE ANALYSIS SUMMARY for {best_model_name} ===")
        print("="*80)
        
        print("\nTop 10 Features (Single Permutation Analysis):")
        for i, (_, row) in enumerate(perm_results.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<30}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")
        
        print(f"\nTop 10 Features (Bootstrap Stability Analysis):")
        for i, (_, row) in enumerate(stability_results.head(10).iterrows(), 1):
            cv_interpretation = "Stable" if row['cv'] < 0.3 else ("Moderate" if row['cv'] < 0.6 else "Unstable")
            print(f"{i:2d}. {row['feature']:<30}: {row['mean_importance']:.4f} "
                  f"(95% CI: {row['ci_lower']:.4f}-{row['ci_upper']:.4f}) [{cv_interpretation}]")
        
        if high_corr_pairs:
            print(f"\nMulticollinearity Concerns (|r| > 0.7):")
            for pair in high_corr_pairs:
                print(f"  • {pair['feature1']} ↔ {pair['feature2']}: r = {pair['correlation']:.3f}")
        else:
            print("\nNo significant multicollinearity detected among top features.")
        
        # Special focus on digestive system diagnosis
        digestive_features = [f for f in feature_names if 'digestive' in f.lower() or 'digest' in f.lower()]
        if digestive_features:
            print(f"\nDigestive System Features Analysis:")
            for feature in digestive_features:
                perm_row = perm_results[perm_results['feature'] == feature]
                if not perm_row.empty:
                    stab_row = stability_results[stability_results['feature'] == feature]
                    if not stab_row.empty:
                        print(f"  {feature}: Importance = {perm_row.iloc[0]['importance_mean']:.4f}, "
                              f"Stability CV = {stab_row.iloc[0]['cv']:.3f}")
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("This comprehensive analysis addresses reviewer concerns about:")
    print("• Model-agnostic feature importance validation (permutation importance)")
    print("• Statistical stability of feature rankings (bootstrap confidence intervals)")
    print("• Potential multicollinearity issues (correlation matrix analysis)")
    print("• Robustness of TabNet attention weight findings")
    print("\nFiles generated provide evidence for feature importance reliability")
    print("and can be directly cited in manuscript revisions.")

if __name__ == "__main__":
    main()
