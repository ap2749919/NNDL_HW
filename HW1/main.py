import utils.mnist_reader as mnist_reader
from models.neural_network import NeuralNetwork
from training.trainer import Trainer
from testing.tester import Tester
from tuning.tuner import Tuner
import numpy as np
import matplotlib.pyplot as plt
import math

import sys
sys.path.append('.')

# Load data
X_train, y_train = mnist_reader.load_mnist('data', kind='train')
X_test, y_test = mnist_reader.load_mnist('data', kind='t10k')


X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0



# Convert labels to one-hot encoding
def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]

y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)


def split_train_validation(X, y, validation_fraction=0.2):
    total_samples = X.shape[0]
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    split_point = int(total_samples * (1 - validation_fraction))
    train_indices = indices[:split_point]
    valid_indices = indices[split_point:]
    return X[train_indices], y[train_indices], X[valid_indices], y[valid_indices]

X_train, y_train, X_valid, y_valid = split_train_validation(X_train, y_train, validation_fraction=0.2)


def search_params():
    # 搜索参数
    tuner = Tuner((X_train, y_train), (X_valid, y_valid))
    _, best_params = tuner.tune()

    best_params = {'hidden_size': 150, 'lr': 0.001, 'reg_lambda': 0.0001}

    
    return best_params

def train_model(search=0):
    if search:
        best_params = search_params()
    else:
        best_params = {'hidden_size': 150, 'lr': 0.001, 'reg_lambda': 0.0001}

    best_hidden_size = best_params['hidden_size']
    best_lr = best_params['lr']
    best_reg_lambda = best_params['reg_lambda']
    model = NeuralNetwork(input_size=784, hidden_size=best_hidden_size, output_size=10)  # Adjust as necessary

    # 使用训练集与验证集绘制loss/auc曲线
    trainer = Trainer(model, (X_train, y_train), (X_valid, y_valid), lr=best_lr, reg_lambda=best_reg_lambda)
    trainer.train(epochs=50)  

    # 使用全部训练集训练模型, 并在测试集上测试
    trainer = Trainer(model, (X_train, y_train), (X_test, y_test), lr=best_lr, reg_lambda=best_reg_lambda)
    trainer.train(epochs=50) 

def test_model():
    # 在测试集上测试
    tester = Tester((X_test, y_test))
    tester.test()


# 对模型的权重进行可视化
def load_model_weights(file_path):
    data = np.load(file_path)
    W1 = data['W1']  # Assuming W1 are the weights from the input layer to the first hidden layer
    return W1

def plot_weights(W):
    # Assume W has shape (input_size, hidden_size), for MNIST input_size might be 784 and you can reshape it to 28x28
    hidden_size = W.shape[1]
    # Set up the grid layout for subplots
    # Choose a square-like shape for the grid
    rows = int(math.sqrt(hidden_size))
    cols = int(np.ceil(hidden_size / rows))
    
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 2, rows * 2))
    # If axes is not an array (when there is only one subplot), make it an array for consistent handling
    if hidden_size == 1:
        axes = np.array([axes])
    
    for i, ax in enumerate(axes.flat):
        if i < hidden_size:
            img = W[:, i].reshape(28, 28)  # Reshaping to 28x28 if MNIST
            ax.imshow(img, cmap='viridis', interpolation='nearest')
            ax.axis('off')
        else:
            ax.axis('off')  # Hide any unused subplots

    # plt.show()
    plt.tight_layout()
    plt.savefig(f'weights.png')
    plt.savefig(f'weights.pdf')
    plt.close()

def save_weights_plot():
    W1 = load_model_weights('best_model_weights.npz')
    plot_weights(W1)

if __name__ == '__main__':
    train_model(search=0)
    test_model()
    save_weights_plot()