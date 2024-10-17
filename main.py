
import pandas as pd 
import random
import matplotlib.pyplot as plt
from rdkit import Chem
from dataset import create_pytorch_geometric_graph_list, create_dataloaders
from models import odorGIN
from train import train_single_epoch, test, train
import torch
from rdkit.Chem import Draw
from torch_geometric.explain import Explanation, Explainer, ModelConfig
from torch_geometric import explain
from explainer import visualize_molecule_explanation

# WIP: embedding space.

if __name__ == '__main__':

    df_path = 'data/Multi-Labelled_Smiles_Odors_dataset.csv'
    odor_df = pd.read_csv(df_path)

    # Molecule Visualization 
    # fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(3, 7))
    # for i in range(20):
    #     m = Chem.MolFromSmiles(odor_df['nonStereoSMILES'][i])
    #     img = Draw.MolToImage(m)
    #     axs[i%5][i%2].imshow(img)
    #     axs[i%5][i%2].set_xticks([])
    #     axs[i%5][i%2].set_yticks([])

    # # plt.xticks([])
    # # plt.yticks([])
    # fig.tight_layout()
    # plt.show()


    X = odor_df.iloc[:, 0]
    classes = odor_df.columns[2:]
    y = odor_df.iloc[:, 2:].values

    graph_list = create_pytorch_geometric_graph_list(X, y)

    batch_size, use_shuffle = 1, True
    train_set, val_set, test_set,  train_loader, val_loader, test_loader = create_dataloaders(graph_list, 0.7, 0.2, 0.1, batch_size)

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
    use_pretrained = False


    model_GIN = odorGIN.OdorGIN(in_channels, hidden_channels, num_layers)
    optimizer = torch.optim.Adam(model_GIN.parameters(), lr=lr, weight_decay=weight_decay)


    # WIP: Hyperparameter tuning
    # train(model_GIN, num_epochs, lr, weight_decay, train_loader, val_loader)


    # Explainer for edge structure
    # Node mask: for each node object
    # Edge mask: for each edge object
    ob_explainer = Explainer(
        model_GIN, 
        algorithm=explain.algorithm.GNNExplainer(),
        explanation_type="model", 
        model_config=ModelConfig(mode="binary_classification", task_level="graph", return_type="probs"),
        node_mask_type="object",
        edge_mask_type="object"
    )

    # random visualizations
    idx = random.sample(range(len(test_set)), 5)
    for i in idx:
      visualize_molecule_explanation(test_set, i, ob_explainer)

    