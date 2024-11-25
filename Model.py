import torch
import torch.nn as nn
import torch.nn.functional as F

class MT_Model(nn.Module):
    def __init__(self, input1_dim=4, input2_dim=5,hidden_dim=128, graph_dim=10):
        super(MT_Model, self).__init__()

        # Partie commune
        self.common_layer = nn.Sequential(
            nn.Linear(input1_dim + input2_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, hidden_dim),
            nn.ReLU()
        )

        # Sortie vectorielle
        self.vector_branch = nn.Sequential(
            nn.Linear(hidden_dim, input2_dim),
            nn.Softmax(dim=-1)
        )

        # Sortie matricielle
        self.graph_branch = nn.Sequential(
            nn.Linear(hidden_dim, graph_dim * graph_dim),
            nn.ReLU()
        )

    def forward(self, input1, input2):
        # Combinaison des deux entrées
        x = torch.cat((input1, input2), dim=-1).float()
        x = self.common_layer(x)

        # Sortie vectorielle
        vector_output = self.vector_branch(x)

        # Sortie matricielle
        graph_flat = self.graph_branch(x)
        graph_matrix = graph_flat.view(-1, 10, 10)
        #device = graph_matrix.device

        # Appliquer les transformations spécifiques aux classes
        #class1_values = torch.matmul(graph_matrix, torch.tensor([0, 1, 2], dtype=torch.float32, device=device))
        #class2_values = torch.matmul(graph_matrix, torch.tensor([0, 3, 6], dtype=torch.float32, device=device))

        #graph_output = torch.stack((class1_values, class2_values), dim=-1)

        graph_output = graph_matrix

        return vector_output, graph_output
