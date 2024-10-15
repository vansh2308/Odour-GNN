# import matplotlib.pyplot as plt
# import pandas as pd 
# from rdkit import Chem
# from rdkit.Chem import Draw
# from PIL import Image


# df_path = 'data/Multi-Labelled_Smiles_Odors_dataset.csv'
# odor_df = pd.read_csv(df_path)


# # for atom in m.GetAtoms():
# #     print(atom.get)

# img = Draw.MolToImage(m)

# plt.imshow(img)
# plt.xticks([])
# plt.yticks([])
# plt.show()

# # # im = Image.open(Draw.MolToImage(m))
# # # im.show()


# ============================================================================
# import packages
import numpy as np
from utils import label_encoding
import pandas as pd 
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader


def get_atom_features(atom, use_chirality = False,  hydrogens_implicit = False) -> np.array: 
    '''
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    '''
    
    permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']
    
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    
    atom_type_enc = label_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)    
    n_heavy_neighbors_enc = label_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    formal_charge_enc = label_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    hybridisation_type_enc = label_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    is_in_a_ring_enc = [int(atom.IsInRing())]
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]

    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
                                    
    if use_chirality == True:
        chirality_type_enc = label_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    
    if hydrogens_implicit == True:
        n_hydrogens_enc = label_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)



def get_bond_features(bond,  use_stereochemistry = True):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """

    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

    bond_type_enc = label_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    
    if use_stereochemistry == True:
        stereo_type_enc = label_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)


if __name__ == '__main__':
    df_path = 'data/Multi-Labelled_Smiles_Odors_dataset.csv'
    odor_df = pd.read_csv(df_path)

    m = Chem.MolFromSmiles(odor_df['nonStereoSMILES'][5])
    atom_features = get_atom_features(m.GetAtoms()[0], True, True)
    bond_feature = get_bond_features(m.GetBonds()[0], True)

    print(bond_feature)
