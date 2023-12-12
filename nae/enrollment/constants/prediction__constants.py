import numpy as np

#Random Forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]  # Number of trees in random forest
max_features = ['sqrt', 'log2']  # Number of features to consider at every split
max_depth = [int(x) for x in np.linspace(10, 110, num=11)] + [None]  # Maximum number of levels in tree
min_samples_split = [2, 5, 10]  # Minimum number of samples required to split a node
min_samples_leaf = [1, 2, 4]  # Minimum number of samples required at each leaf node
bootstrap = [True, False]  # Method of selecting samples for training each tree

param_grid_randomized = \
    {
        'classifier__n_estimators': n_estimators,
        'classifier__max_features': max_features,
        'classifier__max_depth': max_depth,
        'classifier__min_samples_split': min_samples_split,
        'classifier__min_samples_leaf': min_samples_leaf,
        'classifier__bootstrap': bootstrap
    }
