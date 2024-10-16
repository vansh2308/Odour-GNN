
import pandas as pd 
from rdkit import Chem
from dataset import create_pytorch_geometric_graph_list, create_dataloaders
from models import odorGIN
from train import train_single_epoch, test, train
import torch

if __name__ == '__main__':

    df_path = 'data/Multi-Labelled_Smiles_Odors_dataset.csv'
    odor_df = pd.read_csv(df_path)

    m = Chem.MolFromSmiles(odor_df['nonStereoSMILES'][5])


    X = odor_df.iloc[:, 0]
    classes = odor_df.columns[2:]
    y = odor_df.iloc[:, 2:].values

    graph_list = create_pytorch_geometric_graph_list(X, y)

    batch_size, use_shuffle = 1, True
    train_loader, val_loader, test_loader = create_dataloaders(graph_list, 0.7, 0.2, 0.1, batch_size)

    # hyperparamters 
    hidden_channels = 64
    num_layers = 2
    dropout_p = 0
    pooling_type = 'mean'
    in_channels = list(graph_list[0].x.shape)[-1]
    out_channels = 1 
    num_epochs = 50
    lr = 1e-4
    weight_decay = 1e-6


    model_GIN = odorGIN.OdorGIN(in_channels, hidden_channels, num_layers)
    optimizer = torch.optim.Adam(model_GIN.parameters(), lr=lr, weight_decay=weight_decay)

    # train_single_epoch(model_GIN, optimizer, train_loader, mode='train')
    
    # precision, all_preds, all_trgts =  test(model_GIN, test_loader)
    train(model_GIN, num_epochs, lr, weight_decay, train_loader, val_loader)




    # for i, batch in enumerate(train_loader):

    #     X, edge_index, edge_attr, y = batch.x, batch.edge_index, batch.edge_attr, batch.y
    #     print(batch, end="\n")
    #     y_pred = model_GIN(X, edge_index, edge_attr)

    #     break





    




