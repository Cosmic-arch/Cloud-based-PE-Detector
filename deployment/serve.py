import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from io import BytesIO
import json
import os

# Define model architecture (same as your training model)
class MalConv(nn.Module):
    def __init__(self, input_size=2381, hidden_size=128, output_dim=1):
        super(MalConv, self).__init__()
        
        # For EMBER feature vectors, we don't need an embedding layer
        # Instead, we'll use fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_dim)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Remove the channel dimension (batch_size, 1, features) -> (batch_size, features)
        x = x.squeeze(1)
        
        # First layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Second layer
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x

# Load the model
def model_fn(model_dir):
    
    # model_path = os.path.join(model_dir, "deployment/model_final.pt")
    
    # Load model state dict
    model_state = torch.load('model_final_2.pt', map_location=torch.device("cpu"))
    # Initialize model with correct input size
    model = MalConv(input_size=2381)
    
    # Load the state dict
    model.load_state_dict(model_state)
    model.eval()
    return model



# def input_fn(request_body, request_content_type):
#     assert request_content_type == 'application/json'
#     data = json.loads(request_body)['inputs']
#     tensor = torch.tensor(data, dtype=torch.float32)
#     return tensor

def predict_fn(input_tensor, model):
    with torch.no_grad():
        prediction = model(input_tensor)
    return prediction.numpy().tolist()

# def output_fn(prediction, response_content_type):
#     return json.dumps({"predictions": prediction})

