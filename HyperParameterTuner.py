from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class HyperparameterTuner:
    """
    Lightweight wrapper for hyperparameter search.

    Warning:
    Leave search-level n_jobs as 1 or None when tuning models that already
    manage their own threading internally, such as CatBoost, XGBoost, or
    LightGBM. Setting n_jobs=-1 here can oversubscribe CPU threads and cause
    contention or lockups.
    """

    def __init__(self, model, param_space, method='random', cv=5, n_iter=10, scoring=None, n_jobs=1, random_state=42):
        self.model = model
        self.param_space = param_space
        self.method = method
        self.cv = cv
        self.n_iter = n_iter
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state

        self._validate_threading_config()
        
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None # Tracks the cross-validated score of the best model

    def fit(self, X, y):
        # Dictionary dispatch mapping string names to their uninstantiated classes via lambdas
        search_strategies = {
            'grid': lambda: GridSearchCV(
                self.model, self.param_space, cv=self.cv, 
                scoring=self.scoring, n_jobs=self.n_jobs
            ),
            'random': lambda: RandomizedSearchCV(
                self.model, self.param_space, n_iter=self.n_iter, cv=self.cv,
                scoring=self.scoring, n_jobs=self.n_jobs, random_state=self.random_state
            )
        }

        if self.method not in search_strategies:
            raise ValueError(f"Tuning method '{self.method}' not implemented yet.")

        print(f"Starting {self.method} search with {self.cv}-fold CV...")
        
        # Execute the lambda to instantiate the correct Scikit-Learn object
        search = search_strategies[self.method]()
        search.fit(X, y)
        
        # Store results
        self.best_estimator_ = search.best_estimator_
        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_
        
        print(f"Tuning complete. Best {self.scoring if self.scoring else 'default'} score: {self.best_score_:.4f}")
        return self

    def predict(self, X):
        self._check_is_fitted()
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        """
        Crucial for classification tasks requiring ROC/AUC evaluation.
        """
        self._check_is_fitted()
        if hasattr(self.best_estimator_, "predict_proba"):
            return self.best_estimator_.predict_proba(X)
        raise AttributeError(f"The underlying model {type(self.best_estimator_).__name__} does not support predict_proba.")

    def get_best_params(self):
        self._check_is_fitted()
        return self.best_params_

    def _validate_threading_config(self):
        """Fail fast on unsafe nested parallelism settings."""
        if self.n_jobs != -1:
            return

        model_name = type(self.model).__name__
        known_self_threaded_models = {
            'CatBoostClassifier', 'CatBoostRegressor',
            'XGBClassifier', 'XGBRegressor',
            'LGBMClassifier', 'LGBMRegressor',
        }
        likely_thread_controls = ('thread_count', 'nthread', 'num_threads')

        if model_name in known_self_threaded_models or any(hasattr(self.model, attr) for attr in likely_thread_controls):
            raise ValueError(
                f"Unsafe threading configuration detected for {model_name}: "
                "n_jobs=-1 at the search level can cause CPU thread contention. "
                "Use n_jobs=1 or None in HyperparameterTuner and configure model threading separately."
            )
        
    def _check_is_fitted(self):
        """Internal helper to ensure fail-fast behavior if methods are called out of order."""
        if self.best_estimator_ is None:
            raise RuntimeError("You must call .fit() before making predictions or retrieving parameters.")