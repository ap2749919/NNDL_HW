import numpy as np
from models.neural_network import NeuralNetwork
from training.trainer import Trainer

class Tuner:
    def __init__(self, train_data, valid_data):
        self.train_data = train_data
        self.valid_data = valid_data

    def tune(self):
        # Define ranges for hyperparameters
        learning_rates = [0.001, 0.01, 0.1]
        hidden_sizes = [50, 100, 150]
        regularization_strengths = [0.0001, 0.001, 0.01]

        best_val = float('inf')
        best_params = {}
        results = []

        # Grid search over hyperparameters
        for lr in learning_rates:
            for hidden_size in hidden_sizes:
                for reg in regularization_strengths:
                    model = NeuralNetwork(input_size=784, hidden_size=hidden_size, output_size=10)
                    trainer = Trainer(model, self.train_data, self.valid_data, lr=lr, reg_lambda=reg)
                    trainer.train(epochs=30)  # Number of epochs can be adjusted
                    val_loss = trainer.validate()

                    results.append((lr, hidden_size, reg, val_loss))

                    if val_loss < best_val:
                        best_val = val_loss
                        best_params = {'lr': lr, 'hidden_size': hidden_size, 'reg_lambda': reg}
                        print(f'New best model with lr {lr}, hidden_size {hidden_size}, reg {reg}: Validation Loss {val_loss:.4f}')

        print("Best parameters:", best_params)
        self.log_results(results)
        return results, best_params

    def log_results(self, results):
        # Log or save the tuning results
        with open('tuning_results.txt', 'w') as f:
            for result in results:
                f.write(f'LR: {result[0]}, Hidden: {result[1]}, Reg: {result[2]}, Loss: {result[3]:.4f}\n')

