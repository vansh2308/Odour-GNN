import wandb
from models.odorGIN import OdorGIN
from train import train_model


def hyp_tuning(model, in_channels, hidden_channels, num_layers, num_epochs, weight_decay, train_loader, val_loader):
    wandb.init()
    lr = wandb.config['learning_rate']
    # epochs = wandb.config.epochs
    model = OdorGIN(in_channels, hidden_channels, num_layers, 138, wandb.config.dropout, wandb.config['pool_type'])
    train_model(model, num_epochs, lr, weight_decay, train_loader, val_loader)


sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "test_acc"},
    "parameters": {
        "dropout": {"min": 0.0, 'max': 0.2},
        "learning_rate": {"min": 0.0001, "max": 0.001},
        "pool_type": {"values": ['mean', 'max', 'add']},
    },
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="odor-gnn")
    wandb.agent(sweep_id, function=hyp_tuning, count=10)