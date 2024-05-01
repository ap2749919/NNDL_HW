import numpy as np
from models.neural_network import NeuralNetwork
import os
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, train_data, valid_data, lr=0.01, reg_lambda=0.001, lr_decay=0.95):
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.lr_decay = lr_decay
        self.best_loss = float('inf')
        self.best_weights = None
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def train_epoch(self):
        indices = np.arange(self.train_data[0].shape[0])
        np.random.shuffle(indices)
        X_train = self.train_data[0][indices]
        y_train = self.train_data[1][indices]
        total_loss, total_batches = 0, 0
        
        for X_batch, Y_batch in self.get_batches((X_train, y_train), batch_size=32):
            Y_hat = self.model.forward(X_batch)
            loss = self.model.compute_loss(Y_batch, Y_hat) + self.regularization_loss()
            self.model.backward(X_batch, Y_batch, Y_hat)
            total_loss += loss
            total_batches += 1
        average_loss = total_loss / total_batches
        self.train_losses.append(average_loss)

    def regularization_loss(self):
        return 0.5 * self.reg_lambda * (np.sum(self.model.W1 ** 2) + np.sum(self.model.W2 ** 2))

    def get_batches(self, data, batch_size):
        for i in range(0, len(data[0]), batch_size):
            yield data[0][i:i+batch_size], data[1][i:i+batch_size]

    def evaluate_accuracy(self, Y, Y_hat):
        predicted_labels = np.argmax(Y_hat, axis=1)
        true_labels = np.argmax(Y, axis=1)
        return np.mean(predicted_labels == true_labels)

    def validate(self):
        # 验证
        X_val, Y_val = self.valid_data
        Y_hat_val = self.model.forward(X_val)
        val_loss = self.model.compute_loss(Y_val, Y_hat_val) + self.regularization_loss()
        self.val_losses.append(val_loss)
        accuracy = self.evaluate_accuracy(Y_val, Y_hat_val)
        self.val_accuracies.append(accuracy)
        return val_loss
    
    


    def train(self, epochs):
        for epoch in range(epochs):
            self.train_epoch()
            val_loss = self.validate()
            print(f'Epoch {epoch+1}, Validation Loss: {val_loss:.4f}')
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_weights = (self.model.W1.copy(), self.model.b1.copy(), self.model.W2.copy(), self.model.b2.copy())
            self.lr *= self.lr_decay
        self.save_model()
        self.plot_metrics()
    

    def save_model(self):
        # 存储模型权重
        np.savez('best_model_weights.npz', W1=self.best_weights[0], b1=self.best_weights[1], W2=self.best_weights[2], b2=self.best_weights[3])
        print("Best model saved.")

    def plot_metrics(self):
        file_name = f'{100}-{self.lr}-{self.reg_lambda}'

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{file_name}.png')
        plt.savefig(f'{file_name}.pdf')
        plt.close()