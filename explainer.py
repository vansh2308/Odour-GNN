
import random
from rdkit import Chem
import torch
from torch_geometric import explain
# from torch_geometric.explain import Explanation, Explainer, ModelConfig
from pickle import NONE
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
from matplotlib import colors
# from rdkit.Chem.Draw import IPythonConsole
# from IPython.display import SVG



def atom_col(atom_imp):
  col = {}
  threshold = torch.mean(atom_imp)
  for i, imp in enumerate(atom_imp):
    if imp > threshold:
      col[i] = colors.to_rgb("#1BBC9B")
    else:
      col[i] = colors.to_rgb('#D3D3D3')
  return col
    
# Assign colors to edges with higher than average scores
def bond_col(mol, bond_imp, edge_id, threshold=0.5):
  col = {}
  threshold = sum(bond_imp) / len(bond_imp)
  for idx, bond in enumerate(mol.GetBonds()):
      i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
      i, j = min(i, j), max(i, j)
      imp = bond_imp[edge_id.index((i, j))]
      if imp > threshold:
        col[i] = colors.to_rgb("#1BBC9B")
      else:
        col[i] = colors.to_rgb('#D3D3D3')
  return col

# Generate visualization for a molecule
def visualize_molecule_explanation(dataset, graph_idx, ob_explainer):
  data = dataset[graph_idx]
  x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
  ob_explanation = ob_explainer(x, edge_index)
  atom_imp = ob_explanation.node_mask
  bond_imp = []
  edge_idx = []
  # average the two "directional" edges for score of undirected edge
  for i in range(len(ob_explanation.edge_mask)):
    a, b = ob_explanation.edge_index[0][i].cpu(), ob_explanation.edge_index[1][i].cpu()
    if a < b:
      edge_idx.append((a, b))
      temp = NONE
      for i in range(len(edge_index[0])):
        if edge_index[0][i].item() == b.item() and edge_index[1][i].item() == a.item():
          temp = i
          break
      bond_imp.append((ob_explanation.edge_mask[i]+ob_explanation.edge_mask[temp])/2)
  mol = Chem.MolFromSmiles(data.smiles)
  cp = Chem.Mol(mol)
  # generate colors according to importance scores
  highlightAtomColors = atom_col(atom_imp)
  highlightAtoms = list(highlightAtomColors.keys())
  highlightBondColors = bond_col(mol, bond_imp, edge_idx)
  highlightBonds = list(highlightBondColors.keys())

  # draw image
  image_width, image_height = 400, 200
  rdDepictor.Compute2DCoords(cp, canonOrient=True)
  drawer = rdMolDraw2D.MolDraw2DCairo(image_width, image_height)
  drawer.drawOptions().useBWAtomPalette()
  drawer.DrawMolecule(
      cp,
      highlightAtoms=highlightAtoms,
      highlightAtomColors=highlightAtomColors,
      highlightBonds=highlightBonds,
      highlightBondColors=highlightBondColors,
  )
  drawer.FinishDrawing()
  drawer.WriteDrawingText(f"./figures/explainations/{graph_idx}.png")


if __name__ == '__main__':
  pass