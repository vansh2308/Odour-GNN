import numpy as np 
import sys
import pandas as pd 
import random
import matplotlib.pyplot as plt
from rdkit import Chem
from dataset import create_pytorch_geometric_graph_list, create_dataloaders
from models import odorGIN
from train import train_single_epoch, test, train_model
import torch
from rdkit.Chem import Draw
from torch_geometric.explain import ModelConfig
from explainer import Explainer
from torch_geometric import explain
from explaination_visualiser import visualize_molecule_explanation
import argparse

# WIP: add terminal args
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='OdourGNN', description='Predicting molecular odour using GNNs')
    parser.add_argument('-path', '--data-path')
    parser.add_argument('-mv', '--mol_vis', action='store_true')
    parser.add_argument('-pre', '--use_pretrained', action='store_false')
    parser.add_argument('-test', '--test', action='store_true')
    parser.add_argument('-exp', '--explain', action='store_true')

    args = parser.parse_args()

    df_path = args.path if args.data_path else 'data/Multi-Labelled_Smiles_Odors_dataset.csv'
    odor_df = pd.read_csv(df_path)

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
    use_pretrained = args.use_pretrained
    test_mode = args.test
    mol_vis = args.mol_vis


    # Molecule Visualization 
    if mol_vis:
        fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(3, 7))
        for i in range(20):
            m = Chem.MolFromSmiles(odor_df['nonStereoSMILES'][i])
            img = Draw.MolToImage(m)
            axs[i%5][i%2].imshow(img)
            axs[i%5][i%2].set_xticks([])
            axs[i%5][i%2].set_yticks([])

        # plt.xticks([])
        # plt.yticks([])
        fig.tight_layout()
        plt.show()



    model_GIN = odorGIN.OdorGIN(in_channels, hidden_channels, num_layers)
    optimizer = torch.optim.Adam(model_GIN.parameters(), lr=lr, weight_decay=weight_decay)

    if not use_pretrained:
       train_model(model_GIN, num_epochs, lr, weight_decay, train_loader, val_loader, batch_size)
    else:
       model_GIN.load_state_dict(torch.load("wandb/run-20241022_115722-1tq4w6oy/files/gin-best-model.pth", weights_only=True))

    # print(model_GIN.state_dict())

    if test_mode:
        classes = odor_df.columns[2:].to_numpy()
        softmax_cutoff = 0.001

        for data in test_loader:
            model_GIN.eval()
            out = model_GIN(data.x, data.edge_index, data.edge_attr)
            out = (out.squeeze() >= softmax_cutoff).float()

            total_correct = int((out == data.y).sum())
            print('Accuracy: {}'.format(round( total_correct/out.shape[0] ,3)))

            labels = classes[data.y.squeeze().nonzero()]
            preds = classes[out.nonzero()]
            if isinstance(labels, np.ndarray):
                labels = labels.flatten()
            if isinstance(preds, np.ndarray):
                preds = preds.flatten()

            print("Ground Truth: {}".format(labels))
            print("Predictions: {}".format(preds))


    # Explainer for edge structure
    # Node mask: for each node object
    # Edge mask: for each edge object
    if(args.explain):
        ob_explainer = Explainer(
            model_GIN, 
            explanation_type="model", 
            model_config=ModelConfig(mode="multiclass_classification", task_level="graph", return_type="probs"),
        )

        # random visualizations
        idx = random.sample(range(len(test_set)), 25)
        for i in idx:
          visualize_molecule_explanation(test_set, i, ob_explainer)

    