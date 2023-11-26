import torch
import torch.nn as nn
import torch.optim as optim

def read_data(file_path):
    features = []
    targets = []
    with open(file_path, "r") as f:
        all_lines = f.readlines()

        for line in all_lines:
            line = line.strip().split('\t')
            X = list(map(float, line[:-1]))
            y = int(line[-1])
            features.append(X)
            targets.append(y)
    
    return torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.long)


def split_data(features, targets, test_size=100, random_state=None):
    torch.manual_seed(random_state)
    indices = torch.randperm(len(features))
    features_train = features[indices[:-test_size]]
    targets_train = targets[indices[:-test_size]]
    features_test = features[indices[-test_size:]]
    targets_test = targets[indices[-test_size:]]
    
    return features_train, targets_train, features_test, targets_test

def scale_features(features_train, features_test):
    mean = features_train.mean(dim=0)
    std = features_train.std(dim=0)
    features_train_scaled = (features_train - mean) / std
    features_test_scaled = (features_test - mean) / std
    
    return features_train_scaled, features_test_scaled

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


def train_neural_network(model, features_train, targets_train, num_epochs=1000, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        outputs = model(features_train)
        loss = criterion(outputs, targets_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate_model(model, features_test, targets_test):
    with torch.no_grad():
        predictions = torch.argmax(model(features_test), dim=1)
    
    num_correct = torch.sum(predictions == targets_test).item()
    num_testing = len(features_test)
    accuracy = num_correct / num_testing
    
    return num_correct, num_testing, accuracy


features, targets = read_data("D:/Master's work/AI/Pima.txt")

# Split data
features_train, targets_train, features_test, targets_test = split_data(features, targets, test_size=100, random_state=42)

# Scale features
features_train_scaled, features_test_scaled = scale_features(features_train, features_test)

input_size = features_train.shape[1]
output_size = len(torch.unique(targets_train))
neural_network_model = NeuralNetwork(input_size, output_size)
train_neural_network(neural_network_model, features_train_scaled, targets_train, num_epochs=1000, lr=0.01)

num_correct, num_testing, accuracy = evaluate_model(neural_network_model, features_test_scaled, targets_test)
print("No. correct={}, No. testing examples={}, prediction accuracy={} per cent".format(
    num_correct, num_testing, round(accuracy * 100, 2)))
