import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from einops import rearrange
from model import ViT, Transformer, FeedForward, Attention
from data import VideoDataset

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
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

vivit_model = ViT(
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
loss_function = torch.nn.BCEWithLogitsLoss()  # Or any other appropriate loss function
optimizer = torch.optim.Adam(vivit_model.parameters(), lr=1e-4)

# # Training loop
for epoch in range(epochs):
    for batch in train_loader:
        video_tensors = batch['video']  # Ensure this matches the dictionary key used in __getitem__
        midi_labels = batch['midi_labels']
        # For debugging
        print("Input video tensor shape:", video_tensors.shape)

        print(f"Type of video before calling .to(device): {type(video_tensors)}")
        video_tensors = video_tensors.to(device)
        midi_labels = midi_labels.to(device)


        # Forward pass
        outputs = vivit_model(video_tensors)

        # TODO: for each frame take the class with the biggest probability 
        # TODO: fix training loop
        # probabilities = outputs.detach().cpu().numpy()
        # first_class_probabilities = probabilities[0, :, 0]
        # frame_index = 0  # For the first frame
        # frame_probabilities = probabilities[0, frame_index, :].tolist()

        # print("Probabilities for the first class across all frames in the first video:", first_class_probabilities)
        # print("Probabilities for all classes in the first frame of the first video:", frame_probabilities)

        
        # Compute loss
        loss = loss_function(outputs, midi_labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print loss (or store for later analysis)
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')
        print("outputs: ", outputs)

# Save your model
torch.save(vivit_model.state_dict(), 'vivit_piano.pth')

