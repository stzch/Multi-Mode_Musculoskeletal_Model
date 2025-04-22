import torch
import torch.nn as nn

class FiLM(nn.Module):
    def __init__(self, feature_dim, task_embedding_dim):
        super(FiLM, self).__init__()
        # Layers to generate gamma and beta based on the task embedding
        self.gamma_generator = nn.Linear(task_embedding_dim, feature_dim)
        self.beta_generator = nn.Linear(task_embedding_dim, feature_dim)

    def forward(self, x, task_embedding):
        # task_embedding shape: [batch_size, task_embedding_dim]
        # Generate gamma and beta based on task embedding
        print(x.shape,task_embedding.shape)
        gamma = self.gamma_generator(task_embedding)  # shape: [batch_size, feature_dim]
        beta = self.beta_generator(task_embedding)    # shape: [batch_size, feature_dim]
        print(gamma.shape,beta.shape)
        # Ensure gamma and beta match x’s dimensions for broadcasting
      #  gamma = gamma.unsqueeze(1)  # shape: [batch_size, 1, feature_dim]
      #  beta = beta.unsqueeze(1)    # shape: [batch_size, 1, feature_dim]

        # Apply FiLM modulation
        return gamma * x + beta

x=torch.ones(32,512)
task_indicator=torch.ones(32,1)

f1=FiLM(512,1)

y=f1(x,task_indicator)
