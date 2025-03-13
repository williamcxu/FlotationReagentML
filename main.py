import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import featurization as ft


def evaluate_models(X, y):
    """Evaluates different regression models using LOOCV with hyperparameter tuning."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Normalize features

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
            grid_search.fit(X_scaled, y)
            best_model = grid_search.best_estimator_
            best_score = -grid_search.best_score_
        else:
            best_model = model
            best_score = np.mean([
                mean_absolute_error([y[i]], best_model.fit(np.delete(X_scaled, i, axis=0), np.delete(y, i, axis=0)).predict([X_scaled[i]]))
                for i in range(len(y))
            ])

        results[name] = {"model": best_model, "MAE": best_score}
        print(f"Best MAE: {best_score:.4f}")

    # Print full model performance table
    print("\nModel Performance:")
    print("-" * 30)
    for name, result in results.items():
        print(f"{name:22} | MAE = {result['MAE']:.4f}")
    print("-" * 30)

    # Select best model
    best_model_name = min(results, key=lambda x: results[x]["MAE"])
    best_model = results[best_model_name]["model"]
    print(f"\n🏆 Best Model: {best_model_name} with MAE = {results[best_model_name]['MAE']:.4f}")

    return best_model, scaler, results


def save_predictions(smiles_list, predictions, output_path):
    df_out = pd.DataFrame({"SMILES": smiles_list, "Predicted Adsorption Energy": predictions})
    df_out.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")


# Example usage with a water molecule
# mole = Chem.MolFromSmiles("O")  # Water molecule (H2O)
# bob = bag_of_bonds(mole)
# print("Bag of Bonds:", bob)

# SMILES data
smiles_known = ["CCCCCCCC/C=C\\CCCCCCCC(=O)[O-]",
                "C1=CC=C2C(=C1)C(=O)C3=CC(=C(C(=C3C2=O)O)O)S(=O)(=O)[O-]",
                "C1=CC=C(C=C1)C(=O)NO",
                "C(C(=O)O)C(CC(=O)O)(C(=O)O)O",
                "CCCCCCCCCCCCOP(=O)(O)O",
                "CCCCCCCCCCCC(=O)N(C)CC(=O)[O-]",
                "CCCCCCCC(=O)NO",
                "CCCCCC[C@H](C/C=C\\CCCCCCCC(=O)O)O",
                "CNCC(=O)[O-]",
                "[O-][Si](=O)[O-]",
                "C(C(C(=O)O)S(=O)(=O)[O-])C(=O)O",
                "CCCCCCCCCCCCN"]  # Known adsorption energies
functional_head = ["C(=O)[O-]", "S(=O)(=O)[O-]", "C(=O)NO", "C(=O)O", "OP(=O)(O)O", "C(=O)[O-]", "C(=O)NO", "C(=O)O", "C(=O)[O-]",
                   "[O-][Si](=O)[O-]", "S(=O)(=O)[O-]", "N"]
smiles_unknown = ft.read_smiles("qm9.csv", n=10)  # To predict
# smiles_unknown = ["CCCC(=O)[O-]", "CCCS(=O)(=O)[O-]", "CCCC(=O)NO", "CCCCOP(=O)(O)O"]

# Example adsorption energies (replace with real data)
adsorption_energies = np.array([-298.9083581, -189.8056076, -63.76156057, -190.1, -109.7202813, -206.6133479,
                                -74.99093531, -224.4656009, -332.7598172, -240.7, -289.0706311, -241.9092631])

# Generate feature matrices with consistent padding
X_known, X_unknown = ft.generate_bob_feature_matrix(smiles_known, smiles_unknown)

# # Select best model and scaler
# best_model, scaler, model_results = evaluate_models(X_known, adsorption_energies)
#
# # Predict unknown adsorption energies
# X_unknown_scaled = scaler.transform(X_unknown)
# predicted_energies = best_model.predict(X_unknown_scaled)
# best_mol = np.argmin(predicted_energies)
# print("\nBest molecule ", smiles_unknown[best_mol], " with adsorption energy", predicted_energies[best_mol], " kJ/mol")
#
# save_predictions(smiles_unknown, predicted_energies, "results/predictions.csv")
