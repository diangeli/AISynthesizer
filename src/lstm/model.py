import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True, dropout=dropout)  # 1st layer
        self.dropout1 = nn.Dropout(dropout)  
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True, dropout=dropout)  # 2nd layer
        self.dropout2 = nn.Dropout(dropout) 
        self.lstm3 = nn.LSTM(hidden_size2, hidden_size1, batch_first=True)  # 3rd layer
        self.dropout3 = nn.Dropout(dropout)  
        self.fc = nn.Linear(hidden_size1, output_size)  # Fully connected 
    
    def forward(self, x):
        x, _ = self.lstm1(x) 
        x = self.dropout1(x)  
        x, _ = self.lstm2(x)  
        x = self.dropout2(x)  
        x, _ = self.lstm3(x) 
        x = self.dropout3(x) 
        x = self.fc(x[:, -1, :])  
                
        return x  
