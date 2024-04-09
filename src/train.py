import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from einops import rearrange
from model import ViViT, Transformer, FeedForward, Attention
from data import VideoDataset
from utils import create_midi

epochs = 10

# To apply transformations such as resizing and normalization
from torchvision import transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize to the input size expected by the model
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


train_dataset = VideoDataset(video_dir='silent_videos', midi_dir='midis', transforms=transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

vivit_model = ViViT(
    image_size=(224, 224),    # Height and width of input frames
    num_frames=125,           # Total number of frames in each video
    num_classes=88,           # For example, 88 keys on the piano
    dim=1024,                 # Dimensionality of the token/patch embeddings
    depth=6,                  # Number of transformer blocks (depth)
    heads=8,                  # Number of attention heads
    mlp_dim=2048,             # Dimensionality of the feedforward layer
    pool='cls',               # Pooling method ('cls' for class token, 'mean' for mean pooling)
    channels=3,               # Number of channels in the video frames (RGB, so 3)
    dim_head=64,              # Dimensionality of each attention head
    dropout=0.1,              # Dropout rate
    emb_dropout=0.1           # Embedding dropout rate
)

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vivit_model = vivit_model.to(device)

# Define your loss function and optimizer
loss_function = torch.nn.BCEWithLogitsLoss() 
optimizer = torch.optim.Adam(vivit_model.parameters(), lr=1e-4)

# # Training loop
epochs = 1  
threshold = 0.8  

for epoch in range(epochs):
    for batch_idx, batch in enumerate(train_loader):
        video_tensors = batch['video'].to(device)
        midi_labels = batch['midi_labels'].to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = vivit_model(video_tensors)  # Assuming your model instance is named 'vivit_model'
        
        # Moved this after training -> eval
        # probabilities = outputs.detach().cpu().numpy()

        # # Assuming you want to process batch with index 0
        # if batch_idx == 0:  # Example: processing for the first batch
        #     for i in range(probabilities.shape[0]):
        #         # model_outputs = probabilities[i]
        #         for video_idx, video_probs in enumerate(probabilities):
        #             create_midi(video_probs, epoch, batch_idx, video_idx, threshold=0.8)

        loss = loss_function(outputs, midi_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log the loss
        print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Save your model
torch.save(vivit_model.state_dict(), 'vivit_piano_model.pth')