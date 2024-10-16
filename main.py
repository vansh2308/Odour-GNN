
import pandas as pd 
from rdkit import Chem
from dataset import create_pytorch_geometric_graph_list



if __name__ == '__main__':
    df_path = 'data/Multi-Labelled_Smiles_Odors_dataset.csv'
    odor_df = pd.read_csv(df_path)

    m = Chem.MolFromSmiles(odor_df['nonStereoSMILES'][5])


    X = odor_df.iloc[:, 0]
    classes = odor_df.columns[2:]
    y = odor_df.iloc[:, 2:].values

    graph_list = create_pytorch_geometric_graph_list(X, y)
    print(graph_list[5])