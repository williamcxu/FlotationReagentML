import psi4
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

# Periodic table atomic numbers
atomic_numbers = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18
}


def find_functional_head(mol, functional_group):
    """Finds the atoms belonging to the specified functional head, selecting the one closest to the end of SMILES."""
    substructures = mol.GetSubstructMatches(Chem.MolFromSmarts(functional_group))
    if not substructures:
        return []  # No match found
    func_head_atoms = list(substructures[-1])

    # Include hydrogen atoms bonded to the functional head
    func_head_atoms_with_H = list(func_head_atoms)
    for atom_idx in func_head_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1:  # Hydrogen
                func_head_atoms_with_H.append(neighbor.GetIdx())
    return func_head_atoms_with_H  # Select the last occurrence


def get_atomic_positions(mol):
    """Generates 3D coordinates for a molecule and extracts atomic positions."""
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())  # Generate 3D coordinates
    AllChem.MMFFOptimizeMolecule(mol)   # Optimize the geometry using MMFF
    conf = mol.GetConformer()
    positions = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    return positions


def compute_bond_distances(positions):
    """Computes the pairwise interatomic distances."""
    return squareform(pdist(positions))


def bag_of_bonds(mol, functional_group=None):
    """Computes the Bag of Bonds for a molecule. Option to only include specified functional group."""
    mol = Chem.AddHs(mol)  # Add hydrogens explicitly

    positions = get_atomic_positions(mol)
    distance_matrix = compute_bond_distances(positions)

    atoms = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
    bags = {}

    if functional_group is not None:
        functional_head = find_functional_head(mol, functional_group)

        # Construct bags
        for i in functional_head:
            for j in functional_head[functional_head.index(i):]:
                bond_type = "".join(sorted([atoms[i], atoms[j]]))
                Z_i, Z_j = atomic_numbers.get(atoms[i], 0), atomic_numbers.get(atoms[j], 0)
                bond_value = (Z_i * Z_j) / distance_matrix[i, j] if i != j else 0.5 * Z_i ** 2.4
                if bond_type not in bags:
                    bags[bond_type] = []
                bags[bond_type].append(bond_value)
    else:
        # Construct bags
        for i in range(len(atoms)):
            for j in range(i, len(atoms)):
                bond_type = "".join(sorted([atoms[i], atoms[j]]))
                Z_i, Z_j = atomic_numbers.get(atoms[i], 0), atomic_numbers.get(atoms[j], 0)
                bond_value = (Z_i * Z_j) / distance_matrix[i, j] if i != j else 0.5 * Z_i ** 2.4
                if bond_type not in bags:
                    bags[bond_type] = []
                bags[bond_type].append(bond_value)

    # Sort bond values within each bag
    for key in bags:
        bags[key].sort(reverse=True)

    return bags


def generate_bob_feature_matrix(smiles_known, smiles_unknown, functional_groups=None):
    """Generate a consistently padded feature matrix for both known and unknown molecules."""
    molecules_known = [Chem.MolFromSmiles(smiles) for smiles in smiles_known]
    molecules_unknown = [Chem.MolFromSmiles(smiles) for smiles in smiles_unknown]

    if functional_groups is not None:
        if len(functional_groups) != len(smiles_known):
            raise Exception("Functional groups must be specified for all reagents")
        known_bags = [bag_of_bonds(molecules_known[i], functional_groups[i]) for i in range(len(molecules_known))]
        unknown_bags = [bag_of_bonds(mol) for mol in molecules_unknown]
        all_bags = known_bags + unknown_bags
    else:
        all_bags = [bag_of_bonds(mol) for mol in (molecules_known + molecules_unknown)]

    # Identify all bond types across the dataset
    bond_types = sorted(set(bond for bags in all_bags for bond in bags.keys()))

    # Determine the max number of entries for each bond type across both sets
    max_entries_per_bond = {bond: max(len(bags.get(bond, [])) for bags in all_bags) for bond in bond_types}

    # Function to construct feature vectors with consistent padding
    def construct_feature_vectors(bag_list):
        feature_matrix = []
        for bags in bag_list:
            feature_vector = []
            for bond in bond_types:
                bond_values = bags.get(bond, [])
                padded_values = bond_values + [0] * (max_entries_per_bond[bond] - len(bond_values))
                feature_vector.extend(padded_values)
            feature_matrix.append(feature_vector)
        return np.array(feature_matrix)

    # Generate feature matrices with consistent padding
    X_known = construct_feature_vectors(all_bags[:len(molecules_known)])
    X_unknown = construct_feature_vectors(all_bags[len(molecules_known):])

    return X_known, X_unknown


# Define the molecule (SMILES) and basis set
def compute_quantum_properties(smiles, basis='B3LYP/6-31G**'):
    try:
        # Convert SMILES to geometry
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
        xyz = Chem.MolToXYZBlock(mol)

        # Set up Psi4
        psi4.core.set_output_file("output.dat", False)
        psi4.set_memory("4GB")
        psi4.set_options({"scf_type": "df", "puream": True, "basis": basis})

        # Define molecule in Psi4 format
        molecule = psi4.geometry(xyz)

        # Run SCF calculation
        energy, wfn = psi4.energy("B3LYP-D3BJ/6-31G**", return_wfn=True, molecule=molecule)

        # Get HOMO and LUMO energies
        homo_index = wfn.nalpha() - 1
        lumo_index = homo_index + 1
        homo_energy = wfn.epsilon_a().get(homo_index)
        lumo_energy = wfn.epsilon_a().get(lumo_index)
        homo_lumo_gap = lumo_energy - homo_energy

        # Get dipole moment and convert from a.u. to Debye (1 a.u. = 2.541746 Debye)
        dipole_moment = np.linalg.norm(wfn.variable("SCF DIPOLE")) * 2.541746

        # Run ESP calculation for RESP charges
        # psi4.set_options({"esp_fit": True, "charge": 0, "grid_spacing": 0.2})  # Neutral molecule assumed
        # psi4.esp(basis, wfn=wfn)
        # charges = psi4.variable("RESP CHARGES")
        # esp_charges = np.array(charges)
        # negative_surface_area = sum(esp_charges[esp_charges < 0])  # Approximation

        # Estimate molecular volume (based on Van der Waals radii)
        volume = AllChem.ComputeMolVolume(mol)

        return {
            "SMILES": smiles,
            "HOMO energy (Hartree)": homo_energy,
            "LUMO energy (Hartree)": lumo_energy,
            "HOMO-LUMO gap (Hartree)": homo_lumo_gap,
            "Dipole moment (Debye)": dipole_moment,
            # "Negatively charged surface area": negative_surface_area,
            "Molecular volume (Å³)": volume,
        }
    except Exception as e:
        return {"SMILES": smiles, "Error": str(e)}


# Read input SMILES from CSV or TXT file
def read_smiles(file_path, n=None):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        if "smiles" not in df.columns:
            raise ValueError("CSV must contain a column labeled 'smiles'")
        smiles_list = df["smiles"].tolist()
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path, skiprows=1)
        smiles_list = df["smiles"].tolist()
    elif file_path.endswith(".txt"):
        with open(file_path, "r") as file:
            smiles_list = [line.split()[0] for line in file if line.strip()]
    else:
        raise ValueError("Unsupported file format. Use XLSX, CSV, or TXT.")
    # smiles_list = [s.replace(".[Na+].[Na+]", ".[Na+]") for s in smiles_list[:n]]
    return smiles_list[:n]


# Process multiple molecules and save results
def process_and_save(file_path, output_csv="results/quantum_properties.csv", n=-1):
    smiles_list = read_smiles(file_path, n)
    results = [compute_quantum_properties(smiles) for smiles in tqdm(smiles_list)]
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


# input_file = "dft_results.xlsx"
# process_and_save(input_file, n=12)