This code serves as a first attempt at using ML models to screen for and discover molecules that could potentially serve as flotation reagents, particularly for phosphate flotation.

featurization.py contains the code for generating feature vectors. This includes the traditional bag-of-bonds (BOB) representation, a modified BOB representation only including the functional group head that is likely to participate in surface adsorption, and the ability to generate quantum chemical properties (HOMO, LUMO, HOMO-LUMO gap, dipole moment, and molecular volume) using the psi4 package. Data must come in the form of SMILES codes of molecules. The training dataset adsorption energies were generated in-house and can be found in dft_results.xlsx in the data folder. Other data comes from the QM9 dataset (http://quantum-machine.org/datasets/) or can be generated using code in featurization.py.

modeling.py contains code to evaluate different commonly used regression models and determine the best model based on the mean absolute error obtained when performing leave-one-out cross validation.

main.py contains the setup to perform featurization, preprocess the data, evaluate models, and predict adsorption energies.

Packages (beyond standard Python packages) needed for this code:
- psi4 for quantum chemical computations
- scikit-learn for machine learning
- rdkit for molecular representations
- tqdm for the nice progress bar
