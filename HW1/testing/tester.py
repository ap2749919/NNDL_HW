import numpy as np
from models.neural_network import NeuralNetwork

class Tester:
    def __init__(self, test_data, hidden_szie=150, model_path='best_model_weights.npz'):
        self.test_data = test_data
        self.hidden_size = hidden_szie
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        # 加载模型权重
        data = np.load(self.model_path)
        model = NeuralNetwork(input_size=784, hidden_size=self.hidden_size, output_size=10)  # Adjust sizes as needed
        model.W1 = data['W1']
        model.b1 = data['b1']
        model.W2 = data['W2']
        model.b2 = data['b2']
        return model

    def evaluate_accuracy(self):
        X_test, Y_test = self.test_data
        predictions = self.model.forward(X_test)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(Y_test, axis=1)
        accuracy = np.mean(predicted_labels == true_labels)
        return accuracy

    def test(self):
        accuracy = self.evaluate_accuracy()
        print(f'Test Accuracy: {accuracy:.4f}')

