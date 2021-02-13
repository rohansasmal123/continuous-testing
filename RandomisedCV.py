
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb

# Hyper-parameters
params_cv = {
    'colsample_bytree': [0.3, 0.5, 0.8],                # subsample ratio of columns constructing each tree
    'learning_rate': np.arange(0.03, 0.10, step=0.02),  # step size shrinkage 
    'max_depth': np.arange(3, 6),                       # max depth of tree
    'n_estimators': np.arange(400, 600, step=50),       # number of trees used
    'subsample': [0.5, 0.8, 1.0],                       # subsample ratio of training instances
    'min_child_weight': [5, 8, 10, 15],                 # minimum sum of instance weight needed in a child
    'alpha': [2, 3, 4, 6],                              # L2 regularization term
    'gamma': [2, 3, 4, 6],                              # minimum loss reduction for further partition
    'max_delta_step': [1, 2, 3, 4]                      # maximum step each leaf output must be (good for imblanaced data)
}

# Model
xgb_cv_model = xgb.XGBClassifier(objective='binary:logistic', scale_pos_weight=bal, random_state=123)

# Randomized search
xgb_cv_rs = RandomizedSearchCV(
    estimator=xgb_cv_model,
    param_distributions=params_cv,
    n_iter=7,
    scoring='f1',
    n_jobs=-1,
    iid=True,
    cv=5,
    verbose=1
)


print('Randomized search - Parameter tuning')
tic = time.time()
xgb_cv_rs.fit(X_train, y_train)
toc = time.time()
print('Total elapsed time:', (toc-tic)/60, 'minutes.')

# Best parameters
print('Best parameters from random search:\n', xgb_cv_rs.best_params_)

# Results
print('Accuracy score for XGBoost classifier:', accuracy_score(y_test, xgb_cv_rs.best_estimator_.predict(X_test)))
print('ROC AUC score for XGBoost classifier: ', roc_auc_score(y_test, xgb_cv_rs.best_estimator_.predict(X_test)))
print('F1 score for XGBoost classifier:      ', f1_score(y_test, xgb_cv_rs.best_estimator_.predict(X_test)))