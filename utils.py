
import matplotlib.pyplot as plt 


def label_encoding(x, domain:list):
    return [len(domain)-1] if x not in domain else [domain.index(x) ]

def plot_losses(train_losses, val_losses, title=None):
    """Utility method to plot paired loss curves"""
    plt.plot(range(len(val_losses)), val_losses, label='val')
    plt.plot(range(len(train_losses)), train_losses, label='train')
    plt.legend()
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    pass

