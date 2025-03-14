import numpy as np
from joblib import dump, load

from sklearn.preprocessing import StandardScaler

import featurization as ft
import modeling as ml

def smiles_combos():
    # SMILES data
    smiles_known = ft.read_smiles("data/dft_results.xlsx", n=12, stripNa=False)
    smiles_known.remove("[O-][Si](=O)[O-].[Na+].[Na+]")  # psi4 has issues calculating QM properties for this molecule
    functional_head = ["C(=O)[O-]", "S(=O)(=O)[O-]", "C(=O)NO", "C(=O)O", "OP(=O)(O)O", "C(=O)[O-]", "C(=O)NO",
                       "C(=O)O", "C(=O)[O-]", "S(=O)(=O)[O-]", "N"]

    # Generate a variety of different reagent molecules with the same tail but different functional group heads
    functional_groups = ["O", "N", "[O-]", "[N-]", "CO", "C[O-]", "C=N", "CN", "C=[N-]", "C[N-]", "C=CO", "C=C[O-]",
                         "C(=O)O", "C(=O)[O-]", "C=NO", "C(=O)N", "C=NN", "C=N[O-]", "C(=O)[N-]", "C=N[N-]",
                         "C(=O)NO", "C(=O)N[O-]", "OP(=O)(O)O", "OP(=O)(O)[O-]"]
    ft.generate_smiles_combinations(smiles_known, functional_head, functional_groups,
                                    output_csv="data/functional_group_combos.csv")

def bob_feature_models():
    # SMILES data
    smiles_known = ft.read_smiles("data/dft_results.xlsx", n=12, stripNa=True)
    adsorption_energies = np.array([-298.9083581, -189.8056076, -63.76156057, -190.1, -109.7202813, -206.6133479,
                                    -74.99093531, -224.4656009, -332.7598172, -240.7, -289.0706311, -241.9092631])

    smiles_combo = ft.read_smiles("data/functional_group_combos.csv")

    # Generate feature matrices with consistent padding
    X_known, X_combo = ft.generate_bob_feature_matrix(smiles_known, smiles_combo)
    ft.export_features(X_known, "data/feature_bob_combo_dft.csv")
    ft.export_features(X_combo, "data/feature_bob_combo.csv")

    # Load generated feature matrices
    # X_known = ft.load_features("data/feature_bob_dft.csv")
    # X_unknown = ft.load_features("data/feature_bob_qm9.csv")

    # Normalize features and select best model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_known)
    best_model, model_results = ml.evaluate_models_with_random_search(X_scaled, adsorption_energies, n_iter=100)
    dump(best_model, "best_model_bob_combo.joblib")

    # Load chosen best model
    # best_model = load("best_model_bob.joblib")

    # Predict unknown adsorption energies
    X_combo_scaled = scaler.transform(X_combo)
    predicted_energies = best_model.predict(X_combo_scaled)
    best_mol = np.argmin(predicted_energies)
    print("\nBest molecule ", smiles_combo[best_mol], " with adsorption energy", predicted_energies[best_mol], " kJ/mol")

    # Find molecules with predicted energies better than the best known adsorption energy.
    best_known = np.min(adsorption_energies)
    better_indices = np.where(predicted_energies < best_known)[0]
    if len(better_indices) > 0:
        print(f"Molecules with predicted energy better than {best_known} kJ/mol: ")
        for idx in better_indices:
            print(f"Molecule: {smiles_combo[idx]}, Predicted Energy: {predicted_energies[idx]:.6f} kJ/mol")
    else:
        print("No new molecules predicted to be better.")

    ml.save_predictions(smiles_combo, predicted_energies, "results/bob_combo_predictions.csv")

def bob_head_feature_models():
    # SMILES data
    smiles_known = ft.read_smiles("data/dft_results.xlsx", n=12, stripNa=True)
    functional_head = ["C(=O)[O-]", "S(=O)(=O)[O-]", "C(=O)NO", "C(=O)O", "OP(=O)(O)O", "C(=O)[O-]", "C(=O)NO",
                       "C(=O)O", "C(=O)[O-]", "[O-][Si](=O)[O-]", "S(=O)(=O)[O-]", "N"]
    adsorption_energies = np.array([-298.9083581, -189.8056076, -63.76156057, -190.1, -109.7202813, -206.6133479,
                                    -74.99093531, -224.4656009, -332.7598172, -240.7, -289.0706311, -241.9092631])

    smiles_unknown = ft.read_smiles("data/qm9.csv", n=1000)  # To predict
    # smiles_unknown = ["CCCC(=O)[O-]", "CCCS(=O)(=O)[O-]", "CCCC(=O)NO", "CCCCOP(=O)(O)O"]

    # Generate feature matrices with consistent padding
    X_known, X_unknown = ft.generate_bob_feature_matrix(smiles_known, smiles_unknown, functional_head)
    ft.export_features(X_known, "data/feature_bob_head_dft.csv")
    ft.export_features(X_unknown, "data/feature_bob_head_qm9.csv")

    # Load generated feature matrices
    # X_known = ft.load_features("data/feature_bob_dft.csv")
    # X_unknown = ft.load_features("data/feature_bob_qm9.csv")

    # Normalize features and select best model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_known)
    best_model, model_results = ml.evaluate_models_with_random_search(X_scaled, adsorption_energies, n_iter=100)
    dump(best_model, "best_model_bob_head.joblib")

    # Load chosen best model
    # best_model = load("best_model_bob.joblib")

    # Predict unknown adsorption energies
    X_unknown_scaled = scaler.transform(X_unknown)
    predicted_energies = best_model.predict(X_unknown_scaled)
    best_mol = np.argmin(predicted_energies)
    print("\nBest molecule ", smiles_unknown[best_mol], " with adsorption energy", predicted_energies[best_mol], " kJ/mol")

    # Find molecules with predicted energies better than the best known adsorption energy.
    best_known = np.min(adsorption_energies)
    better_indices = np.where(predicted_energies < best_known)[0]
    if len(better_indices) > 0:
        print(f"Molecules with predicted energy better than {best_known} kJ/mol: ")
        for idx in better_indices:
            print(f"Molecule: {smiles_unknown[idx]}, Predicted Energy: {predicted_energies[idx]:.6f} kJ/mol")
    else:
        print("No new molecules predicted to be better.")

    ml.save_predictions(smiles_unknown, predicted_energies, "results/bob_head_predictions.csv")

def qm_feature_models():
    # SMILES data
    smiles_known = ft.read_smiles("data/dft_results.xlsx", n=12, stripNa=False)
    smiles_known.remove("[O-][Si](=O)[O-].[Na+].[Na+]") # psi4 has issues calculating QM properties for this molecule
    functional_head = ["C(=O)[O-]", "S(=O)(=O)[O-]", "C(=O)NO", "C(=O)O", "OP(=O)(O)O", "C(=O)[O-]", "C(=O)NO",
                       "C(=O)O", "C(=O)[O-]", "S(=O)(=O)[O-]", "N"]
    adsorption_energies = np.array([-298.9083581, -189.8056076, -63.76156057, -190.1, -109.7202813, -206.6133479,
                                    -74.99093531, -224.4656009, -332.7598172, -289.0706311, -241.9092631])

    # To predict
    smiles_combos = ft.read_smiles("data/functional_group_combos.csv")
    smiles_unknown = ft.read_smiles("data/qm9.csv", n=1000)

    # Generate QM properties for all molecules (using same software for consistency)
    # ft.qm_properties(smiles_known, output_csv="data/qm_properties_known.csv")
    # ft.qm_properties(smiles_combos, output_csv="data/qm_properties_combos.csv")
    # ft.qm_properties(smiles_unknown, output_csv="data/qm_properties_qm9.csv")

    # Load generated feature matrices
    X_known = ft.load_features("data/qm_properties_known.csv", header=0, index_col=0)
    # X_combos = ft.load_features("data/qm_properties_combos.csv", header=0, index_col=0)
    # X_unknown = ft.load_features("data/qm_properties_qm9.csv", header=0, index_col=0)

    # Normalize features and select best model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_known)
    best_model, model_results = ml.evaluate_models_with_random_search(X_scaled, adsorption_energies, n_iter=100)
    dump(best_model, "best_model_qm.joblib")

if __name__ == "__main__":
    smiles_combos()

    # qm_feature_models()

    bob_feature_models()