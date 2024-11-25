import os
import matplotlib.pyplot as plt
import torch
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader


class Trainer:
    def __init__(self, model, data, dataset_folder, train_size = 0.8, learning_rate=0.01, batch_size=16, num_epochs=700, gpu=True):
        self.device = torch.device('cuda' if gpu else 'cpu')
        print(f"Using device: {self.device}")
        self.model = model.to(self.device)
        self.data = data
        self.dataset_folder = dataset_folder
        self.train_size = train_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # Critère basé sur la divergence KL
        self.criterion = nn.KLDivLoss(reduction="batchmean")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        print("Training the model ")

        Traine_files = int(self.train_size * len(os.listdir(self.dataset_folder)))
        Test_files = len(os.listdir(self.dataset_folder)) - Traine_files
        print("Training files : ", Traine_files, "Testing files : ", Test_files)

        train_losses = []
        test_losses = []

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0

            for file in range(1, Traine_files + 1):
                input1, input2, graph, tj = self.data.get_dataset(file)

                input1_tensor = torch.tensor(input1, dtype=torch.float64)
                input2_tensor = torch.tensor(input2, dtype=torch.float64)

                dataset = TensorDataset(input1_tensor, input2_tensor)
                train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


                for batch_input1, batch_input2 in train_loader:
                    batch_input1, batch_input2 = batch_input1.to(self.device), batch_input2.to(self.device)

                    self.optimizer.zero_grad()
                    vector_output, _ = self.model(batch_input1, batch_input2)

                    # la divergence KL
                    log_vector_output = torch.log(vector_output + 1e-8)  # Éviter log(0)
                    loss = self.criterion(log_vector_output, batch_input2)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()

                avg_train_loss = running_loss / len(train_loader)
                train_losses.append(avg_train_loss)

            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {avg_train_loss:.4f}", end=" ")

            # Évaluation sur le jeu de test
            self.model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for file in range(Traine_files+1, Test_files + 1):
                    input1, input2, graph, tj = self.data.get_dataset(file)
                    input1_tensor = torch.tensor(input1, dtype=torch.float64)
                    input2_tensor = torch.tensor(input2, dtype=torch.float64)
                    dataset = TensorDataset(input1_tensor, input2_tensor)
                    test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

                    for batch_input1, batch_input2 in test_loader:
                        batch_input1, batch_input2 = batch_input1.to(self.device), batch_input2.to(self.device)
                        vector_output, _ = self.model(batch_input1, batch_input2)
                        log_vector_output = torch.log(vector_output + 1e-8)
                        loss = self.criterion(log_vector_output, batch_input2)
                        test_loss += loss.item()

                    avg_test_loss = test_loss / len(test_loader)
                    test_losses.append(avg_test_loss)
                    print(f"Test Loss: {avg_test_loss:.4f}")

        self.plot_losses(train_losses, test_losses)

    def predict(self, input1, input2):
        self.model.eval()
        with torch.no_grad():
            input1 = input1.clone().detach().float().to(self.device)
            input2 = input2.clone().detach().float().to(self.device)
            vector_output, graph_output = self.model(input1, input2)
        return vector_output, graph_output

    def plot_losses(self, train_losses, test_losses, output_dir="output"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Testing Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "losses.png"))
        plt.close()


    def plot_accuracy(self, train_accuracy, test_accuracy, output_dir="output"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.figure()
        plt.plot(train_accuracy, label="Train Accuracy")
        plt.plot(test_accuracy, label="Test Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Testing Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "accuracy.png"))
        plt.close()

