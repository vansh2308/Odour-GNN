
import pandas as pd 
from rdkit import Chem
from dataset import create_pytorch_geometric_graph_list, create_dataloaders



if __name__ == '__main__':
    df_path = 'data/Multi-Labelled_Smiles_Odors_dataset.csv'
    odor_df = pd.read_csv(df_path)

    m = Chem.MolFromSmiles(odor_df['nonStereoSMILES'][5])


    X = odor_df.iloc[:, 0]
    classes = odor_df.columns[2:]
    y = odor_df.iloc[:, 2:].values

    graph_list = create_pytorch_geometric_graph_list(X, y)

    batch_size, use_shuffle = 32, True
    train_loader, val_loader, test_loader = create_dataloaders(graph_list, 0.7, 0.2, 0.1, batch_size)

    # for batch in train_loader:
    #     print(batch)
    #     break




