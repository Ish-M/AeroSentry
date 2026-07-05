import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn

class AeroSentryLSTM(nn.Module):
    def __init__(self, sequence_length=30, num_features=17, hidden_dim=64, num_layers=2):
        super(AeroSentryLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Deep LSTM network to process temporal correlations
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,   # Expects shape: (batch_size, seq_length, num_features)
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        # Dense regression layers to handle final RUL calculation
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 1) # Final scalar RUL output
        
    def forward(self, x):
        # Initialize hidden and cell states to zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward pass through LSTM layers
        out, _ = self.lstm(x, (h0, c0))
        
        # Pull the final hidden state output of the sequence (last cycle step)
        out = out[:, -1, :]
        
        # Process through fully connected regression layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

if __name__ == '__main__':
    model = AeroSentryLSTM()
    test_tensor = torch.randn(16, 30, 17)
    output = model(test_tensor)
    print(f"[+] LSTM Compiled. Input {test_tensor.shape} -> Output shape {output.shape}")