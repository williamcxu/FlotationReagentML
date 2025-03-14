import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneOut, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from scipy.stats import uniform, randint, loguniform

def evaluate_models(X, y):
    """Evaluates different regression models using LOOCV with hyperparameter tuning."""
    models = {
        "Linear Regression": (LinearRegression(), {}),
        "Ridge Regression": (Ridge(), {"alpha": [0.01, 0.1, 1, 10, 100]}),
        "Lasso Regression": (Lasso(), {"alpha": [0.01, 0.1, 1, 10, 100]}),
        "Gaussian Process": (GaussianProcessRegressor(), {
            "alpha": [1e-10, 1e-5, 1e-2, 0.1],
            "kernel": [ConstantKernel(1.0) * RBF(length_scale) for length_scale in [0.1, 1, 10]]
        }),
        "Random Forest": (RandomForestRegressor(), {
            "n_estimators": [10, 50, 100, 200],
            "max_depth": [None, 5, 10]
        })
    }

    loo = LeaveOneOut()
    results = {}

    for name, (model, param_grid) in models.items():
        print(f"\nTraining {name}...")

        if param_grid:
            grid_search = GridSearchCV(model, param_grid, cv=loo, scoring="neg_mean_absolute_error", n_jobs=-1)
            grid_search.fit(X, y)
            best_model = grid_search.best_estimator_
            best_score = -grid_search.best_score_
            best_params = grid_search.best_params_
        else:
            # For linear regression, no hyperparameter tuning needed, just manual LOOCV
            predictions = []

            for i in range(len(y)):
                X_train = np.delete(X, i, axis=0)
                y_train = np.delete(y, i, axis=0)
                X_test = X[i].reshape(1, -1)

                best_model = model.fit(X_train, y_train)
                pred = best_model.predict(X_test)[0]
                predictions.append(pred)

            best_score = mean_absolute_error(y, predictions)
            best_params = "N/A"

        results[name] = {"model": best_model, "MAE": best_score, "Hyperparameters": best_params}
        print(f"Best MAE: {best_score:.4f}")
        print(f"Best Hyperparameters: {best_params}")

    # Print full model performance table
    print("\nModel Performance:")
    print("-" * 30)
    print(f"{'Model':22} | {'MAE':8}")
    print("-" * 30)
    for name, result in results.items():
        print(f"{name:22} | MAE = {result['MAE']:.4f}")
        print(f"\t{result['Hyperparameters']}")
    print("-" * 30)

    # Select best model
    best_model_name = min(results, key=lambda x: results[x]["MAE"])
    best_model = results[best_model_name]["model"]
    print(f"\n🏆 Best Model: {best_model_name} with MAE = {results[best_model_name]['MAE']:.4f}")

    return best_model, results

def evaluate_models_with_random_search(X, y, n_iter=20, random_state=42):
    """Evaluates different regression models using RandomizedSearchCV with hyperparameter tuning."""
    # Define hyperparameter grids for each model
    models = {
        "Linear Regression": (LinearRegression(), {}),
        "Ridge Regression": (Ridge(), {
            "alpha": loguniform(0.001, 100)  # Sample alpha with a log-uniform distribution
        }),
        "Lasso Regression": (Lasso(), {
            "alpha": loguniform(0.001, 100)  # Lasso alpha with log-uniform
        }),
        "ElasticNet": (ElasticNet(), {
            "alpha": loguniform(0.001, 100),
            "l1_ratio": uniform(0, 1)
        }),
        "SVR": (SVR(), {
            "C": loguniform(0.1, 100),
            "epsilon": uniform(0.01, 1),
            "kernel": ["linear", "rbf", "poly"]  # Fixed list
        }),
        "Gaussian Process": (GaussianProcessRegressor(), {
            "alpha": loguniform(1e-10, 1e-1),  # Noise regularization alpha
            "kernel": [ConstantKernel(1.0, (0.1, 10)) * RBF(length_scale, (0.1, 10))
                       for length_scale in loguniform(0.1, 10).rvs(5)],  # Sample 5 random kernels
        }),
        "Random Forest": (RandomForestRegressor(), {
            "n_estimators": randint(50, 200),  # Random number of trees
            "max_depth": randint(3, 30),  # Random tree depth
            "min_samples_split": randint(2, 11)
        }),
        "Gradient Boosting": (GradientBoostingRegressor(), {
            "n_estimators": randint(50, 200),
            "learning_rate": loguniform(0.001, 0.1),
            "max_depth": randint(3, 20)
        }),
        "Decision Tree": (DecisionTreeRegressor(), {
            "max_depth": randint(3, 30),
            "min_samples_split": randint(2, 11)
        }),
        "KNN Regressor": (KNeighborsRegressor(), {
            "n_neighbors": randint(1, 50),
            "weights": ["uniform", "distance"]
        }),
        "MLP Regressor": (MLPRegressor(), {
            "hidden_layer_sizes": [(randint(50, 200).rvs(),),
                                   (randint(50, 100).rvs(), randint(20, 50).rvs())],
            "alpha": loguniform(0.0001, 0.1),
            "learning_rate": ["constant", "adaptive"]
        })
    }

    loo = LeaveOneOut()
    results = {}

    # Run RandomizedSearchCV for each model
    for name, (model, param_grid) in models.items():
        print(f"\nTraining {name} with RandomizedSearchCV...")

        if param_grid:  # If the model has hyperparameters to tune
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
                cv=loo,  # Perform cross-validation with 5 folds
                random_state=random_state
            )
            random_search.fit(X, y)
            best_model = random_search.best_estimator_
            predictions = best_model.predict(X)
            best_params = random_search.best_params_
            best_score = -random_search.best_score_  # Convert negative MAE back to positive

        else:
            # For linear regression, no hyperparameter tuning needed, just manual LOOCV
            predictions = []

            for i in range(len(y)):
                X_train = np.delete(X, i, axis=0)
                y_train = np.delete(y, i, axis=0)
                X_test = X[i].reshape(1, -1)

                best_model = model.fit(X_train, y_train)
                pred = best_model.predict(X_test)[0]
                predictions.append(pred)

            best_params = "N/A"
            best_score = mean_absolute_error(y, predictions)

        # Evaluate R² score on the whole dataset
        r2 = r2_score(y, predictions)

        # Store results
        results[name] = {
            "model": best_model,
            "MAE": best_score,
            "R²": r2,
            "Hyperparameters": best_params
        }

        print(f"{name} Best MAE: {best_score:.4f}")
        print(f"{name} R²: {r2:.4f}")
        print(f"{name} Best Hyperparameters: {best_params}")

    # Print overall results in a performance table
    print("\nModel Performance Summary:")
    print("-" * 50)
    print(f"{'Model':22} | {'MAE':8} | {'R²':8}")
    print("-" * 50)
    for name, result in results.items():
        print(f"{name:22} | {result['MAE']:.4f} | {result['R²']:.4f}")
        print(f"\t{result['Hyperparameters']}")
    print("-" * 50)

    # Return best models
    best_model_name = min(results, key=lambda x: results[x]["MAE"])
    best_model = results[best_model_name]["model"]
    print(f"\n🏆 Best Model: {best_model_name} with MAE = {results[best_model_name]['MAE']:.4f}")
    print(f"Best Hyperparameters: {results[best_model_name]['Hyperparameters']}")

    return best_model, results


def save_predictions(smiles_list, predictions, output_path):
    df_out = pd.DataFrame({"SMILES": smiles_list, "Predicted Adsorption Energy": predictions})
    df_out.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
