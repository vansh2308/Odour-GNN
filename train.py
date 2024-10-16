import torch 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
from utils import plot_losses
from sklearn.metrics import f1_score, roc_auc_score


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
        out = model(data.x, data.edge_index, data.batch)
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






def train(model, num_epochs, lr, weight_decay, train_loader, val_loader):
    '''
    Train or finetune the given model using the train/val sets
    '''

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # track metrics
    roc_scores, f1_scores, accs, losses, val_losses = [], [], [], [], []

    for epoch in tqdm(range(num_epochs)):
        loss = train_single_epoch(model, optimizer, train_loader, 'train')
        val_loss = train_single_epoch(model, optimizer, val_loader, 'val')
        train_acc, train_preds, train_true  = test(model, train_loader)
        test_acc, test_preds, test_true = test(model, val_loader)

        # calculate bootstrapped ROC AUC score over entire val set
        _, whole_val_preds, whole_val_true = test(model, val_loader)
        whole_val_preds = whole_val_preds.squeeze()
        whole_val_true = whole_val_true.squeeze()
        rocauc_score = roc_auc_score(whole_val_true.cpu(), whole_val_preds.cpu())
        f1 = f1_score(whole_val_true.cpu(), whole_val_preds.cpu(), average='weighted')

        # track metrics
        roc_scores.append(rocauc_score)
        f1_scores.append(f1)
        accs.append(test_acc)
        losses.append(loss)
        val_losses.append(val_loss)

    best_f1_score = max(f1_scores)
    best_f1_epoch = f1_scores.index(best_f1_score)
    best_auc_score = max(roc_scores)
    best_auc_epoch = roc_scores.index(best_auc_score)

    plot_losses(losses, val_losses, title='finetuning: train vs. val loss')
    fig, ax = plt.subplots(1, 2, figsize=(6, 2))
    for i, (name, metric) in enumerate([
        ('f1', f1_scores), ('roc', roc_scores)
    ]):
        ax[i].plot(range(len(metric)), metric)
        ax[i].set_ylim((0, 1))
        ax[i].set_title(name)
    plt.show()
    
    return best_f1_score, best_f1_epoch, best_auc_score, best_auc_epoch




if __name__ == '__main__':
    pass