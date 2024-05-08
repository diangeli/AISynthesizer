import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from mido import MidiFile, MidiTrack, Message, MetaMessage
from model import Classifier  
import re


def natural_keys(text):
    return [int(c) if c.isdigit() else c for c in re.split('(\d+)', text)]

def process_video_to_midi(video_path, model, device, output_dir):
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
    
    active_notes = {}  # Dictionary to keep track of active notes and their off times
    frame_index = 0  
    current_tick = 0
    note_count = 0


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
            # print(note)
            note_count += 1
            if note not in active_notes:
                track.append(Message('note_on', note=note, velocity=64, time=ticks_per_note))
                track.append(Message('note_off', note=note, velocity=0, time=ticks_per_note))
                active_notes[note] = current_tick + ticks_per_note  # Schedule note off
        
        current_tick = 1  
        frame_index += 1

    track.append(MetaMessage('end_of_track'))
    midi_filename = os.path.join(output_dir, os.path.basename(video_path).replace('.mp4', ''))
    midi.save(midi_filename)
    cap.release()
    print(f"MIDI file has been saved as '{midi_filename}'.")

def process_all_videos(video_dir, checkpoint_name, checkpoints_root, output_base_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_folder = os.path.join(checkpoints_root, checkpoint_name)
    checkpoint_file = os.path.join(model_folder, 'model.pt')
    if os.path.exists(checkpoint_file):
        print(checkpoint_file)
        model = Classifier(num_classes=91, encoder='resnet18', pretrained=True)
        model.load_state_dict(torch.load(checkpoint_file, map_location=device))
        model.to(device)
        model.eval()

        output_dir = os.path.join(output_base_dir, checkpoint_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        video_files = sorted(video_files, key=natural_keys)  # Sort files numerically

        for video_file in video_files:
            video_path = os.path.join(video_dir, video_file)
            process_video_to_midi(video_path, model, device, output_dir)
    else:
        print(f"No model file found in {model_folder}")

video_dir = 'silent_videos'
checkpoints_root = 'dimos_checkpoints'
checkpoint_name = 'ResNet18'  
output_base_dir = 'classification_midis/ResNet18'
process_all_videos(video_dir, checkpoint_name, checkpoints_root, output_base_dir)