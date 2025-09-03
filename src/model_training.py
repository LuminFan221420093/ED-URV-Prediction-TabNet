import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
from pytorch_tabnet.tab_model import TabNetClassifier
import warnings

warnings.filterwarnings('ignore')

# Load data
file_path = 'C:/Users/CWang/HuXiaoLi/TrainingCohortStandardData.xlsx'
data = pd.read_excel(file_path)

# Define features and labels
X = data.drop(columns=['Group']).values
y = data['Group'].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define models
log_reg = LogisticRegression(max_iter=1000)
svm_model = svm.SVC(probability=True)
random_forest = RandomForestClassifier()
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
tabnet = TabNetClassifier()  # Initialize TabNet classifier

# 5-fold stratified cross-validation with 1 repetition
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)

'''
# Define parameter grids
param_grid_lr = {
    'C': [0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear']
}

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [4, 5, 6],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]
}

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [6, 8, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
'''

# Define parameter grids (optimized for smaller sample size)

param_grid_rf = {
    'n_estimators': [50, 100],  # Reduce number of trees to lower complexity
    'max_features': ['sqrt'],  # Maintain feature randomness to avoid overfitting
    'max_depth': [4, 6],  # Reduce tree depth
    'min_samples_split': [5, 10],  # Control minimum samples in internal nodes to prevent overfitting
    'min_samples_leaf': [2, 4]  # Control minimum samples in leaf nodes
}

param_grid_lr = {
    'C': [0.01, 0.1, 1],
    'penalty': ['l1','l2'],
    'solver': ['liblinear'],
    'tol': [1e-4, 1e-5]  # Adjust convergence tolerance
}

param_grid_svm = {
    'C': [0.01, 0.1, 1],  # Lower C value range
    'kernel': ['rbf', 'sigmoid'],  # Keep common non-linear kernels, remove poly to avoid overfitting
    'gamma': ['scale','auto'],  # 'scale' is more stable for small datasets
     'tol': [1e-3, 1e-4]  # Set different tolerance levels for convergence conditions
}

param_grid_xgb = {
    'n_estimators': [50, 100, 150],  # Reduce number of trees to prevent overfitting
    'learning_rate': [0.1, 0.2, 0.3],  # Higher learning rate for faster convergence
    'max_depth': [3, 4, 5],  # Lower tree depth, suitable for small datasets
    'min_child_weight': [1, 3],  # Control minimum samples in leaf nodes to prevent overfitting
    'gamma': [0, 0.1],  # Keep minimal gamma values to reduce model complexity
    'subsample': [0.7, 1]  # Control sample subset ratio for each tree to enhance generalization
}

'''
param_grid_knn = {
    'n_neighbors': [3, 5],
    'weights': ['uniform'],
    'metric': ['euclidean']
}
'''

# Define function to plot learning curves
def plot_learning_curves(grid_search, model_name):
    results = grid_search.cv_results_
    mean_train_score = results['mean_train_score']
    mean_test_score = results['mean_test_score']
    std_train_score = results['std_train_score']
    std_test_score = results['std_test_score']
    
    param_combinations = range(len(mean_train_score))
    best_index = grid_search.best_index_
    
    # Plot training and validation accuracy curves
    plt.figure(figsize=(9, 6))
    plt.plot(param_combinations, mean_train_score, label='Training Accuracy', color='orange')
    plt.fill_between(param_combinations,
                     mean_train_score - std_train_score,
                     mean_train_score + std_train_score, color='orange', alpha=0.2)
    
    plt.plot(param_combinations, mean_test_score, label='Validation Accuracy', color='skyblue')
    plt.fill_between(param_combinations,
                     mean_test_score - std_test_score,
                     mean_test_score + std_test_score, color='skyblue', alpha=0.2)
    
    # Mark best model position
    plt.axvline(x=best_index, color='k', linestyle='--', label='Best Model')
    plt.title(f'{model_name} Training & Validation Accuracy')
    plt.xlabel('Hyperparameter Set Index')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{model_name}_learning_curve_accuracy.jpg', format='jpg', dpi=300)
    plt.show()

    # Plot training and validation loss curves
    train_loss = 1 - mean_train_score
    test_loss = 1 - mean_test_score
    
    plt.figure(figsize=(9, 6))
    plt.plot(param_combinations, train_loss, label='Training Loss', color='red')
    plt.fill_between(param_combinations,
                     train_loss - std_train_score,
                     train_loss + std_train_score, color='red', alpha=0.2)
    
    plt.plot(param_combinations, test_loss, label='Validation Loss', color='blue')
    plt.fill_between(param_combinations,
                     test_loss - std_test_score,
                     test_loss + std_test_score, color='blue', alpha=0.2)
    
    # Mark best model position
    plt.axvline(x=best_index, color='k', linestyle='--', label='Best Model')
    plt.title(f'{model_name} Training & Validation Loss')
    plt.xlabel('Hyperparameter Set Index')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{model_name}_learning_curve_loss.jpg', format='jpg', dpi=300)
    plt.show()

# Perform grid search and save best model and parameters
def save_best_model_and_params(grid_search, model_name):
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    dump(best_model, f'Best{model_name}Model.joblib')
    
    # Save parameters to Excel file
    params = best_model.get_params(deep=True)
    params_df = pd.DataFrame(list(params.items()), columns=['Parameter', 'Value'])
    params_df.to_excel(f'Best{model_name}ModelParameters.xlsx', index=False)
    
    # Save cross-validation results to Excel file
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.to_excel(f'Best{model_name}ModelCVResults.xlsx', index=False)
    
    # Plot learning curves
    plot_learning_curves(grid_search, model_name)

    return best_model

'''
# Perform grid search and save best model and parameters
def save_best_model_and_params(grid_search, model_name):
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    dump(best_model, f'Best{model_name}Model.joblib')
    
    # Save parameters to Excel file
    params = best_model.get_params(deep=True)
    params_df = pd.DataFrame(list(params.items()), columns=['Parameter', 'Value'])
    params_df.to_excel(f'Best{model_name}ModelParameters.xlsx', index=False)
    
    # Save cross-validation results to Excel file
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.to_excel(f'Best{model_name}ModelCVResults.xlsx', index=False)
    
    return best_model
    '''

'''grid_search_rf = GridSearchCV(estimator=random_forest, param_grid=param_grid_rf, cv=cv, n_jobs=1, verbose=2)
best_rf = save_best_model_and_params(grid_search_rf, 'RandomForest')
'''
grid_search_rf = GridSearchCV(estimator=random_forest, param_grid=param_grid_rf, cv=cv, n_jobs=1, verbose=2, return_train_score=True)
best_rf = save_best_model_and_params(grid_search_rf, 'RandomForest')

'''grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=cv, n_jobs=1, verbose=2)
best_xgb = save_best_model_and_params(grid_search_xgb, 'XGBoost')
'''
grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=cv, n_jobs=1, verbose=2, return_train_score=True)
best_xgb = save_best_model_and_params(grid_search_xgb, 'XGBoost')

'''grid_search_lr = GridSearchCV(estimator=log_reg, param_grid=param_grid_lr, cv=cv, n_jobs=1, verbose=2)
best_lr = save_best_model_and_params(grid_search_lr, 'LogisticRegression')
'''
grid_search_lr = GridSearchCV(estimator=log_reg, param_grid=param_grid_lr, cv=cv, n_jobs=1, verbose=2, return_train_score=True)
best_lr = save_best_model_and_params(grid_search_lr, 'LogisticRegression')

'''grid_search_svm = GridSearchCV(estimator=svm_model, param_grid=param_grid_svm, cv=cv, n_jobs=1, verbose=2)
best_svm = save_best_model_and_params(grid_search_svm, 'SVM')
'''

grid_search_svm = GridSearchCV(estimator=svm_model, param_grid=param_grid_svm, cv=cv, n_jobs=1, verbose=2, return_train_score=True)
best_svm = save_best_model_and_params(grid_search_svm, 'SVM')

import torch
import matplotlib.pyplot as plt
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, log_loss

# Set global font size
plt.rcParams.update({'font.size': 16})

# Data loading section omitted, assuming X, y are already available

# Use KFold for TabNet cross-validation training, adjustable fold number
kf = KFold(n_splits=5, random_state=42, shuffle=True)
best_accuracy = 0
best_tabnet_model = None
best_fold = 0
best_epoch = 0
best_loss = None

fold_idx = 1  # Initialize fold index

for train_index, test_index in kf.split(X):
    X_train_fold, X_val_fold = X[train_index], X[test_index]
    y_train_fold, y_val_fold = y[train_index], y[test_index]

    # Initialize TabNet classifier
    model = TabNetClassifier(
        n_d=8,  # Decision dimension
        n_a=8,  # Attention dimension
        n_steps=4,  # Decision steps
        gamma=1.3,  # Gamma parameter
        momentum=0.3,  # Momentum
        lambda_sparse=1e-3,  # Sparse regularization
        optimizer_fn=torch.optim.Adam,  # Use Adam optimizer
        optimizer_params=dict(lr=0.01)  # Set learning rate
    )

    # Train TabNet model
    model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_train_fold, y_train_fold), (X_val_fold, y_val_fold)],
        eval_name=['Train', 'Validation'],
        eval_metric=['accuracy', 'logloss'],
        max_epochs=200,  # Maximum training epochs
        patience=30,  # Early stopping
        batch_size=32,  # Batch size
        virtual_batch_size=16,  # Virtual batch size
        drop_last=False  # Whether to drop last batch
    )
    
    # Evaluate validation set performance
    y_val_pred = model.predict(X_val_fold)
    val_accuracy = accuracy_score(y_val_fold, y_val_pred)
    y_val_pred_proba = model.predict_proba(X_val_fold)
    val_loss = log_loss(y_val_fold, y_val_pred_proba)

    # Check if this is the best model, directly use model.best_epoch
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_tabnet_model = model
        best_loss = val_loss
        best_fold = fold_idx
        best_epoch = model.best_epoch  # Use TabNetClassifier provided attribute

    print(f"Fold {fold_idx} - Loss: {val_loss}, Accuracy: {val_accuracy}")
    fold_idx += 1  # Update fold index

# Report best model detailed information
print(f"Best TabNet Model - Fold: {best_fold}, Epoch: {best_epoch}, Loss: {best_loss}, Accuracy: {best_accuracy}")

# Save best TabNet model
save_path = 'BestTabNetModel'
best_tabnet_model.save_model(save_path)

# Store best models in dictionary
models = {'Logit': best_lr, 'RF': best_rf, 'SVM': best_svm, 'XGBoost': best_xgb, 'TabNet': best_tabnet_model}

#models = {'LASSO': best_lr, 'RF': best_rf, 'SVM': best_svm, 'XGBoost': best_xgb}

#models = {'XGBoost': best_xgb}

# Extract all metrics from model history
train_loss = best_tabnet_model.history['Train_logloss']
validation_loss = best_tabnet_model.history['Validation_logloss']
train_accuracy = best_tabnet_model.history['Train_accuracy']
validation_accuracy = best_tabnet_model.history['Validation_accuracy']
epochs = range(1, len(train_loss) + 1)  # Epochs start from 1

# Plot training loss and validation loss in same figure
plt.figure(figsize=(9, 6))
plt.plot(epochs, train_loss, 'r-', label='Training Loss')
plt.plot(epochs, validation_loss, 'b-', label='Validation Loss')
plt.axvline(x=best_epoch + 1, color='k', linestyle='--', label='Best Model Epoch')  # Add 1 to best_epoch to match x-axis starting from 1
plt.title(f'Training & Validation Loss (Best Model from Fold {best_fold})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_and_validation_loss.jpg', format='jpg', dpi=300)
plt.show()

# Calculate y-axis range for accuracy
accuracy_min = min(min(train_accuracy), min(validation_accuracy))
accuracy_max = max(max(train_accuracy), max(validation_accuracy))
accuracy_range = accuracy_max - accuracy_min
accuracy_ylim_lower = accuracy_min - accuracy_range * 0.5
accuracy_ylim_upper = accuracy_max + accuracy_range * 0.5

# Plot training accuracy and validation accuracy in same figure
plt.figure(figsize=(9, 6))
plt.plot(epochs, train_accuracy, 'orange', label='Training Accuracy')
plt.plot(epochs, validation_accuracy, 'skyblue', label='Validation Accuracy')
plt.axvline(x=best_epoch + 1, color='k', linestyle='--', label='Best Model Epoch')  # Similarly adjust dashed line position
plt.title(f'Training & Validation Accuracy (Best Model from Fold {best_fold})')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(accuracy_ylim_lower, accuracy_ylim_upper)
plt.legend()
plt.savefig('training_and_validation_accuracy.jpg', format='jpg', dpi=300)
plt.show()

from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, confusion_matrix
# Evaluate model performance and plot ROC curves
plt.figure(figsize=(7, 7))

for model_name, model in models.items():
    if model_name == 'TabNet':
        y_pred_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Reference', alpha=.8)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('1-Specificity', fontsize=16, labelpad=10)
plt.ylabel('Sensitivity', fontsize=16, labelpad=10)
plt.title('ROC Curves', fontsize=18, pad=15)
plt.legend(loc='lower right', fontsize=16)
plt.tight_layout()
plt.xticks(np.arange(0, 1.1, 0.1), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=14)
plt.savefig('optimized_roc_curves.jpg', dpi=300)
plt.show()

from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score  # Ensure these functions are imported
# Calculate and print evaluation metrics for each model
for model_name, model in models.items():
    y_pred = model.predict(X_test) if model_name == 'TabNet' else model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\n{model_name} Model:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import mean_squared_error
# Calibration curve analysis
plt.figure(figsize=(7, 7))
brier_scores = {}

for model_name, model in models.items():
    print(f"Evaluating {model_name} (Calibration Curve):")

    # Calibration Curve calculations
    prob_true, prob_pred = [], []
    for train_idx, val_idx in StratifiedKFold(n_splits=5).split(X_train, y_train):
        # Fit the model on the training part of the fold
        if model_name != 'TabNet':
            model.fit(X_train[train_idx], y_train[train_idx])
        
        # Predict probabilities on the validation part of the fold
        y_pred_prob = model.predict_proba(X_train[val_idx])[:, 1] if model_name != 'TabNet' else model.predict_proba(X_train[val_idx])[:, 1]
        
        # Calculate the calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(y_train[val_idx], y_pred_prob, n_bins=7)

        # Interpolate the calibration curve for uniform x-values
        prob_true.append(np.interp(np.linspace(0, 1, 20), mean_predicted_value, fraction_of_positives))
        prob_pred.append(np.linspace(0, 1, 20))

    # Average the calibration curve over all folds
    mean_prob_true = np.mean(prob_true, axis=0)
    mean_prob_pred = np.mean(prob_pred, axis=0)
    plt.plot(mean_prob_pred, mean_prob_true, label=model_name, marker='o', markersize=4)

    # Calculate Brier score on the mean calibration curve
    brier_score = mean_squared_error(mean_prob_true, mean_prob_pred)
    brier_scores[model_name] = brier_score
    print(f"Brier Score for {model_name}: {brier_score:.4f}")

# Calibration curve formatting
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Reference', alpha=.8)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Mean Predicted Probability', fontsize=14, labelpad=20)
plt.ylabel('Fraction of Positives', fontsize=14, labelpad=20)
plt.title('Calibration Curves', fontsize=16, y=1.05)
plt.legend(loc='lower right')

# Set x and y axis tick ranges and intervals
plt.xticks(np.arange(0, 1.1, 0.1), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=14)

# Modify legend function, add prop parameter to set label font size
plt.legend(loc='lower right', fontsize=14, prop={'size': 14})

# Adjust figure position
plt.subplots_adjust(left=0.15)  # Increase left margin

# Save image as JPG format, 300dpi
plt.savefig('Calibration_Curves.jpg', format='jpg', dpi=300, bbox_inches='tight')
plt.show()

# Decision curve analysis
def calculate_net_benefit(thresh_group, y_pred_prob, y_label):
    net_benefit = []
    for thresh in thresh_group:
        y_pred_label = y_pred_prob > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit.append(tp / n - fp / n * (thresh / (1 - thresh)))
    return net_benefit

def plot_dca(ax, thresh_group, net_benefit_model, color, model_name):
    ax.plot(thresh_group, net_benefit_model, color=color, lw=2, label=model_name)
    max_nb_idx = np.argmax(net_benefit_model)
    if net_benefit_model[max_nb_idx] > net_benefit_all[max_nb_idx]:
        ax.fill_between(thresh_group, net_benefit_model, net_benefit_all, color=color, alpha=0.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, max(net_benefit_model) + 0.1)
    ax.set_xlabel('Threshold Probability', fontsize=22, fontweight='normal', labelpad=20)
    ax.set_ylabel('Net Benefit', fontsize=22, fontweight='normal', labelpad=20)
    ax.set_title('Decision Curve Analysis', fontsize=24, fontweight='normal', y=1.05)
    ax.grid(True, alpha=0.5)
    ax.legend(loc='upper right', fontsize=24)
    return ax

thresh_group = np.arange(0, 1.01, 0.01)
net_benefit_all = np.zeros_like(thresh_group)
tn, fp, fn, tp = confusion_matrix(y_test, np.ones_like(y_test)).ravel()
n = len(y_test)
net_benefit_all = tp / n - fp / n * (thresh_group / (1 - thresh_group))

fig, ax = plt.subplots(figsize=(12, 10))
ax.plot(thresh_group, net_benefit_all, color='gray', lw=2, linestyle='--', label='Treat All')
ax.plot([0, 1], [0, 0], color='gray', lw=2, linestyle=':', label='Treat None')

colors = ['blue', 'orange', 'green', 'red', 'purple']
for i, (model_name, model) in enumerate(models.items()):
    y_pred_prob = model.predict_proba(X_test)[:, 1] if model_name == 'TabNet' else model.predict_proba(X_test)[:, 1]
    net_benefit_model = calculate_net_benefit(thresh_group, y_pred_prob, y_test)
    ax = plot_dca(ax, thresh_group, net_benefit_model, color=colors[i], model_name=model_name)
    print(f"Optimal threshold for {model_name}: {thresh_group[np.argmax(net_benefit_model)]:.2f}")
    print(f"DCA score for {model_name}: {np.max(net_benefit_model):.3f}\n")

plt.xticks(np.arange(0, 1.1, 0.1), fontsize=20)
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=20)
plt.show()

fig.savefig("Decision_Curve_Analysis.jpg", dpi=300, bbox_inches='tight')

import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(grid_search, model_name):
    results = grid_search.cv_results_
    mean_train_score = results['mean_train_score']
    mean_test_score = results['mean_test_score']
    std_train_score = results['std_train_score']
    std_test_score = results['std_test_score']
    
    params = results['params']
    best_index = grid_search.best_index_
    best_train_score = mean_train_score[best_index]
    best_test_score = mean_test_score[best_index]
    
    # Plotting Accuracy Curve
    plt.figure(figsize=(9, 6))
    plt.plot(range(len(mean_train_score)), mean_train_score, label='Training Accuracy', color='orange')
    plt.fill_between(range(len(mean_train_score)),
                     mean_train_score - std_train_score,
                     mean_train_score + std_train_score, color='orange', alpha=0.2)
    
    plt.plot(range(len(mean_test_score)), mean_test_score, label='Validation Accuracy', color='skyblue')
    plt.fill_between(range(len(mean_test_score)),
                     mean_test_score - std_test_score,
                     mean_test_score + std_test_score, color='skyblue', alpha=0.2)
    
    plt.axvline(x=best_index, color='k', linestyle='--', label='Best Model')
    plt.title(f'{model_name} Training & Validation Accuracy')
    plt.xlabel('Hyperparameter Set Index')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plotting Loss Curve
    # Note: GridSearchCV does not directly provide loss, but we can derive it using 1-accuracy
    train_loss = 1 - mean_train_score
    test_loss = 1 - mean_test_score
    
    plt.figure(figsize=(9, 6))
    plt.plot(range(len(train_loss)), train_loss, label='Training Loss', color='red')
    plt.fill_between(range(len(train_loss)),
                     train_loss - std_train_score,
                     train_loss + std_train_score, color='red', alpha=0.2)
    
    plt.plot(range(len(test_loss)), test_loss, label='Validation Loss', color='blue')
    plt.fill_between(range(len(test_loss)),
                     test_loss - std_test_score,
                     test_loss + std_test_score, color='blue', alpha=0.2)
    
    plt.axvline(x=best_index, color='k', linestyle='--', label='Best Model')
    plt.title(f'{model_name} Training & Validation Loss')
    plt.xlabel('Hyperparameter Set Index')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Plot learning curves for each model
plot_learning_curves(grid_search_rf, 'RF')
plot_learning_curves(grid_search_xgb, 'XGBoost')
plot_learning_curves(grid_search_lr, 'LR')
plot_learning_curves(grid_search_svm, 'SVM')

import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(grid_search, model_name):
    results = grid_search.cv_results_
    mean_train_score = results['mean_train_score']
    mean_test_score = results['mean_test_score']
    std_train_score = results['std_train_score']
    std_test_score = results['std_test_score']
    
    best_index = grid_search.best_index_
    
    # Dynamic y-axis limits for accuracy
    min_accuracy = min(np.min(mean_train_score), np.min(mean_test_score))
    max_accuracy = max(np.max(mean_train_score), np.max(mean_test_score))
    accuracy_margin = (max_accuracy - min_accuracy) * 0.3  # Add 30% margin for better visualization
    
    # Plotting Accuracy Curve
    plt.figure(figsize=(9, 6))
    plt.plot(range(len(mean_train_score)), mean_train_score, label='Training Accuracy', color='orange')
    plt.plot(range(len(mean_test_score)), mean_test_score, label='Validation Accuracy', color='skyblue')
    
    plt.axvline(x=best_index, color='k', linestyle='--', label='Best Model')
    plt.title(f'{model_name} Training & Validation Accuracy')
    plt.xlabel('Hyperparameter Set Index')
    plt.ylabel('Accuracy')
    plt.ylim(min_accuracy - accuracy_margin, max_accuracy + accuracy_margin)  # Set dynamic y-limits
    plt.legend()
    plt.savefig(f'{model_name}_learning_curve_accuracy.jpg', format='jpg', dpi=300)  # Save as JPG
    plt.show()

    # Dynamic y-axis limits for loss
    train_loss = 1 - mean_train_score
    test_loss = 1 - mean_test_score
    min_loss = min(np.min(train_loss), np.min(test_loss))
    max_loss = max(np.max(train_loss), np.max(test_loss))
    loss_margin = (max_loss - min_loss) * 0.3  # Add 30% margin for better visualization
    
    # Plotting Loss Curve
    plt.figure(figsize=(9, 6))
    plt.plot(range(len(train_loss)), train_loss, label='Training Loss', color='red')
    plt.plot(range(len(test_loss)), test_loss, label='Validation Loss', color='blue')
    
    plt.axvline(x=best_index, color='k', linestyle='--', label='Best Model')
    plt.title(f'{model_name} Training & Validation Loss')
    plt.xlabel('Hyperparameter Set Index')
    plt.ylabel('Loss')
    plt.ylim(min_loss - loss_margin, max_loss + loss_margin)  # Set dynamic y-limits
    plt.legend()
    plt.savefig(f'{model_name}_learning_curve_loss.jpg', format='jpg', dpi=300)  # Save as JPG
    plt.show()

# Plot learning curves for each model
plot_learning_curves(grid_search_rf, 'RF')
plot_learning_curves(grid_search_xgb, 'XGBoost')
plot_learning_curves(grid_search_lr, 'LR')
plot_learning_curves(grid_search_svm, 'SVM')
