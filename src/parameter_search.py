from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV


def source_best_params(X_train, y_train):

    clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    param_grid = {
        'n_estimators': [300],
        'max_depth': [15, 17, 20, 23, 25],
        'learning_rate': [0.015],
        'subsample': [0.9],
        'colsample_bytree': [0.8],
        'min_child_weight': [0.5],
        'gamma': [0.7],
        'reg_lambda': [0.13],
        'reg_alpha':[1e-5]}

    rs_clf = RandomizedSearchCV(
        clf, param_grid, n_iter=20,
        n_jobs=1, verbose=2, cv=3,
        scoring='neg_log_loss', refit=False,
        random_state=42
    )

    print('Randomized Search for Best Parameters...')
    rs_clf.fit(X_train, y_train)
    
    print('Optimal hyperparameters have been identified.')

    best_score = rs_clf.best_score_
    best_params = rs_clf.best_params_

    print('Best Score: ', best_score)
    print('Best Params: ')
    for param_name in best_params.keys():
        print(param_name, best_params[param_name])

    return best_params
