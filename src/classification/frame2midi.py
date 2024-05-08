import cv2
import torch
import numpy as np
from torchvision import transforms
from mido import MidiFile, MidiTrack, Message, MetaMessage
from mido import bpm2tempo
from model import Classifier

def process_video_to_midi(video_path, model, device):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        return

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    track.append(MetaMessage('set_tempo', tempo=500000)) 
    ticks_per_note = 480  
    
    current_tick = 0
    active_notes = {}
    note_count = 0
    frame_index = 0  


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = transform(frame).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            output = model(frame_tensor)
            predicted = torch.sigmoid(output).cpu().numpy()

        threshold = 0.9
        current_active_notes = set(np.where(predicted >= threshold)[1] + 17) 
        for note in current_active_notes:
            print(note)
            note_count += 1
            if note not in active_notes:
                track.append(Message('note_on', note=note, velocity=64, time=ticks_per_note))
                track.append(Message('note_off', note=note, velocity=0, time=ticks_per_note))
                active_notes[note] = current_tick + ticks_per_note  # Schedule note off
        
        current_tick = 1  
        frame_index += 1

    track.append(MetaMessage('end_of_track'))
    midi.save('output.mid')
    cap.release()
    print(f"MIDI file has been saved as 'output.mid'. Total frames processed: {frame_index}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier(num_classes=91, encoder='resnet18', pretrained=True)
model.load_state_dict(torch.load('dimos_checkpoints/ResNet18_pt/model.pt', map_location=device))
model.to(device)
model.eval()

video_path = 'silent_videos/track_111.mid.mp4'
process_video_to_midi(video_path, model, device)
