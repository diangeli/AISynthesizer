import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from einops import rearrange
from model import ViViT, Transformer, FeedForward, Attention
from dataset import VideoDataset
from utils import create_midi
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from torchvision import transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = VideoDataset(video_dir='silent_videos', midi_dir='midis', transforms=transform, mode='train', limit=None)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

vivit_model = ViViT(
    image_size=(224, 224),    # Height and width of input frames
    num_frames=125,           # Total number of frames in each video
    num_classes=88,           # For example, 88 keys on the piano
    dim=1024,                 # Dimensionality of the token/patch embeddings
    depth=6,                  # Number of transformer blocks (depth)
    heads=8,                  # Number of attention heads
    mlp_dim=2048,             # Dimensionality of the feedforward layer
    pool='mean',               # Pooling method ('cls' for class token, 'mean' for mean pooling)
    channels=3,               # Number of channels in the video frames (RGB, so 3)
    dim_head=64,              # Dimensionality of each attention head
    dropout=0.05,              # Dropout rate
    emb_dropout=0.05           # Embedding dropout rate
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vivit_model = vivit_model.to(device)
optimizer = torch.optim.Adam(vivit_model.parameters(), lr=1e-4)
loss_function = torch.nn.BCELoss()

lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15, verbose=True)

early_stopping_patience = 5 
best_loss = float('inf')  
epochs_without_improvement = 0  

# Training loop
epochs = 2  # Set a reasonable number of epochs

for epoch in range(epochs):
    epoch_loss = 0  # Track the total loss for this epoch

    for batch_idx, batch in enumerate(train_loader):
        video_tensors = batch['video'].to(device)
        midi_labels = batch['midi_labels'].to(device)

        optimizer.zero_grad()  # Clear gradients
        outputs = vivit_model(video_tensors)  # Forward pass

        loss = loss_function(outputs, midi_labels)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        epoch_loss += loss.item()  # Accumulate loss

        # Log the loss
        print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Calculate the average loss for this epoch
    epoch_loss_avg = epoch_loss / len(train_loader)

    # Learning rate scheduler update
    lr_scheduler.step(epoch_loss_avg)

    # Early stopping check
    if epoch_loss_avg < best_loss:
        best_loss = epoch_loss_avg
        epochs_without_improvement = 0  # Reset the counter
    else:
        epochs_without_improvement += 1

    # If early stopping condition is met, break the training loop
    if epochs_without_improvement >= early_stopping_patience:
        print("Early stopping triggered")
        break

# Save the trained model
torch.save(vivit_model.state_dict(), 'checkpoints/vivit_piano_model_new.pth')