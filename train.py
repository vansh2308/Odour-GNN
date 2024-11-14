import torch 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
import os 
import wandb
from utils import plot_losses
from sklearn.metrics import f1_score, roc_auc_score
# from torch_geometric.nn import save_pretrained


# WIP: Hyperparameter tuning, save best model, 

softmax_cutoff = 0.005

def train_single_epoch(model, optimizer, train_loader, mode):
    '''
    Train model for one epoch
    '''

    if mode == "train":
        model.train()
    elif mode == "val" or mode == "test":
        model.eval()

    total_loss = 0
    for data in train_loader:
        # data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = torch.nn.CrossEntropyLoss()(out.squeeze(), data.y)

        if mode == "train":
            loss.backward()
            optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader.dataset)  



@torch.no_grad()
def test(model, loader):
    '''
    Test the model on test loader 
    '''

    model.eval()
    total_correct = 0
    total_ex = 0 
    all_preds = []
    all_true = [] 
    for data in loader:
        # data = data.to(device)

        pred = model(data.x, data.edge_index, data.batch)
        pred = (pred.squeeze() >= softmax_cutoff).float()


        total_correct += int((pred == data.y).sum()) 
        total_ex += np.prod(data.y.shape) 

        all_preds.append(pred)
        all_true.append(data.y)
    return total_correct / total_ex, torch.vstack(all_preds), torch.vstack(all_true)





def train_model(model, num_epochs, lr, weight_decay, train_loader, val_loader, batch_size):
    '''
    Train or finetune the given model using the train/val sets
    '''

    run = wandb.init(project="OdorGNN")
    wandb.config = {"epochs": 5, "learning_rate": lr, "batch_size": batch_size}

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_accs, test_accs, train_losses, val_losses = [], [], [], []

    for epoch in tqdm(range(num_epochs)):
        train_loss = train_single_epoch(model, optimizer, train_loader, 'train')
        val_loss = train_single_epoch(model, optimizer, val_loader, 'val')
        train_acc, train_preds, train_true  = test(model, train_loader)
        test_acc, test_preds, test_true = test(model, val_loader)

        wandb.log({"Train Accuracy": train_acc, "Train Loss": train_loss, "Val Accuracy": test_acc, "Val Loss": val_loss})
        
        # track metrics
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # torch.save(model.state_dict(), 'gin-best-model.pth')

    torch.save(model.state_dict(),  os.path.join(wandb.run.dir, "gin-best-model.pth"))
    # wandb.save('model.h5')
    # wandb.save('../logs/*ckpt*')
    # wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))

    artifact = wandb.Artifact('gin-best-model', type='model')
    # artifact.add_file('gin-best-model.pth')
    # run.log_artifact(artifact)

    run.finish()
    return train_losses, val_losses, train_accs, test_accs



if __name__ == '__main__':
    pass