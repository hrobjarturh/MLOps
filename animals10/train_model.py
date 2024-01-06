import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from animals10.models import GoogLeNet


class Trainer:
    def __init__(self, model, device, criterion, optimizer) -> None:
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

    def load(self, filepath="/data/processed/dataset.pt", local=True):
        if local:
            try:
                dataset = torch.load(filepath)
            except FileNotFoundError:
                print(f"Error: File '{filepath}' not found.")
                exit()

            self.data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        else:
            # TODO: Load data with dvs
            self.data_loader = None

    def train(self, num_epochs=2):
        for epoch in range(num_epochs):
            self.model.train()

            for inputs, labels in self.data_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.logits, labels)
                loss.backward()
                self.optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")  # TODO: logger

    def validate(self, val_loader):  # TODO: Create validation set
        self.model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    def save_model(self, filepath="models/googlenet_model.pth"):
        torch.save(self.model.state_dict(), filepath)


if __name__ == "__main__":
    model = GoogLeNet().model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(model, device, criterion, optimizer)
