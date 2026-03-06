import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, make_scorer, f1_score,
    precision_score, recall_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from scipy.stats import randint, uniform


import warnings
warnings.filterwarnings('ignore')

def LR(df):
    #print(f"Loaded matrix: {df.shape}")
    #print(f"  Label=1 (AD): {(df['LABEL']==1).sum()}")
    #print(f"  Label=0 (Control): {(df['LABEL']==0).sum()}")

    # Separate features and target
    feature_cols = [c for c in df.columns if c not in ['SUBJECT_ID', 'LABEL']]
    X = df[feature_cols]
    y = df['LABEL']

    #hit memory limits so convert to sparse matrix and float32. Shouldn't lose much precision since these are mostly binary features.
    X = X.astype(np.float32)
    X_sparse = csr_matrix(X.values)


    #print(f"Features: {X.shape[1]}")


    N_FOLDS = 5
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    scoring = {
        'accuracy': 'accuracy',
        'f1': make_scorer(f1_score, zero_division=0),
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'roc_auc': 'roc_auc',
    }

    #print(f"Cross-validation: {N_FOLDS}-fold stratified")
    #print(f"Positive class prevalence: {y.mean():.4f}")

    
    # Lasso needs scaled features for proper regularization
    scaler = StandardScaler(with_mean=False)  # with_mean=False because X is sparse
    X_scaled = scaler.fit_transform(X_sparse)

    '''
    lasso = LogisticRegression(
        penalty='l1',
        solver='saga',
        class_weight='balanced',
        max_iter=5000,
        C=1.0,               # Regularization strength (lower = more regularization)
        random_state=42,
        n_jobs=1 #changing to 1 to help with memory issues on large datasets
    )
    '''

    lasso = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        class_weight='balanced',
        max_iter=5000,
        C=1.0,               # Regularization strength (lower = more regularization)
        random_state=42,
    )

    #print("Training Lasso (L1 Logistic Regression) with cross-validation...")
    lasso_cv = cross_validate(
        lasso, X_scaled, y,
        cv=skf,
        scoring=scoring,
        return_estimator=True,
        n_jobs=1 # changing to 1 to help with memory issues on large datasets
    )

    #print("\nLasso Results (mean +/- std across folds):")
    for metric in scoring:
        scores = lasso_cv[f'test_{metric}']
        #print(f"  {metric}: {scores.mean():.4f} +/- {scores.std():.4f}")

    # Average absolute coefficients across folds
    lasso_coefs = np.mean(
        [np.abs(est.coef_[0]) for est in lasso_cv['estimator']],
        axis=0
    )

    lasso_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': lasso_coefs
    }).sort_values('importance', ascending=False)

    #print(f"\nNon-zero coefficient features: {(lasso_coefs > 0).sum()}")

    return lasso_cv, lasso_importance_df

def RF(df):
    #print(f"Loaded matrix: {df.shape}")
    #print(f"  Label=1 (AD): {(df['LABEL']==1).sum()}")
    #print(f"  Label=0 (Control): {(df['LABEL']==0).sum()}")

    # Separate features and target
    feature_cols = [c for c in df.columns if c not in ['SUBJECT_ID', 'LABEL']]
    X = df[feature_cols]
    y = df['LABEL']

    #hit memory limits so convert to sparse matrix and float32. Shouldn't lose much precision since these are mostly binary features.
    X = X.astype(np.float32)
    X_sparse = csr_matrix(X.values)


    #print(f"Features: {X.shape[1]}")


    N_FOLDS = 5
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    scoring = {
        'accuracy': 'accuracy',
        'f1': make_scorer(f1_score, zero_division=0),
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'roc_auc': 'roc_auc',
    }

    #print(f"Cross-validation: {N_FOLDS}-fold stratified")
    #print(f"Positive class prevalence: {y.mean():.4f}")

    ''' Add Random Forest'''
    # setup random forest classifier with hyperparameters that are reasonable for this dataset #this is first pass
    # class_weight='balanced_subsample' reweights per bootstrap sample

    rf_base = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=5,
        min_samples_split=10,
        max_features='sqrt',        # decorrelate trees
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1,
    )

    #print("\n[1/3] Training Random Forest (baseline) with cross-validation...")
    rf_cv = cross_validate(
        rf_base, X_sparse, y,
        cv=skf,
        scoring=scoring,
        return_estimator=True,
        n_jobs=1  # outer loop sequential to avoid mem issues w/ nested parallelism
    )

    #print("\nRandom Forest Baseline Results (mean ± std across folds):")
    for metric in scoring:
        scores = rf_cv[f'test_{metric}']
        #print(f"  {metric}: {scores.mean():.4f} ± {scores.std():.4f}")


    # now do hyperparameter tuning with RandomizedSearchCV
    #print("\n[2/3] Running RandomizedSearchCV for hyperparameter tuning...")

    param_dist = {
        'n_estimators': randint(300, 2000),
        'max_depth': randint(5, 25),
        'min_samples_leaf': randint(2, 20),
        'min_samples_split': randint(2, 20),
        'max_features': ['sqrt', 'log2', 0.1, 0.2, 0.3],
        'class_weight': ['balanced', 'balanced_subsample'],
    }

    optimzation_metric = 'recall'  # prioritize recall so we never miss a patient

    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=30,
        scoring=optimzation_metric,  # prioritize recall so we never miss a patient
        cv=skf,
        random_state=42,
        n_jobs=1,
        verbose=1,
    )
    rf_search.fit(X_sparse, y)

    #print(f"\nBest {optimzation_metric} from search: {rf_search.best_score_:.4f}")
    #print(f"Best params: {rf_search.best_params_}")


    # cross val
    #print("\n[3/3] Evaluating tuned Random Forest with cross-validation...")
    rf_tuned = rf_search.best_estimator_

    rf_tuned_cv = cross_validate(
        rf_tuned, X_sparse, y,
        cv=skf,
        scoring=scoring,
        return_estimator=True,
        n_jobs=1
    )

    #print("\nTuned Random Forest Results (mean ± std across folds):")
    for metric in scoring:
        scores = rf_tuned_cv[f'test_{metric}']
        #print(f"  {metric}: {scores.mean():.4f} ± {scores.std():.4f}")



    # Average feature importances across folds
    rf_tuned_importances = np.mean(
        [est.feature_importances_ for est in rf_tuned_cv['estimator']],
        axis=0
    )
    rf_tuned_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_tuned_importances
    }).sort_values('importance', ascending=False)

    return rf_cv, rf_tuned_cv, rf_tuned_importance_df