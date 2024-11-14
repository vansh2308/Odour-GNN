
import matplotlib.pyplot as plt 


def label_encoding(x, domain:list):
    return [len(domain)-1] if x not in domain else [domain.index(x) ]

def plot_losses(train_losses, val_losses,train_accs, val_accs, title=None):
    """Utility method to plot paired loss curves"""
    # print(train_accs, val_accs)
    plt.plot(range(len(val_losses)), val_losses, label='val')
    plt.plot(range(len(train_losses)), train_losses, label='train')
    plt.plot(range(len(train_accs)), train_accs, label='train_acc')
    plt.plot(range(len(val_accs)), val_accs, label='val_acc')
    plt.legend()
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    pass

