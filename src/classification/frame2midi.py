import cv2
import torch
import numpy as np
import pretty_midi
from torchvision import transforms

from model import Classifier

def process_video_to_midi(video_path, model, device):
    # transformations for each frame
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    midi_file = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)

    # Set tempo
    tempo = 500000  # microseconds per beat, corresponds to 120 BPM
    midi_file.tempo_changes.append(pretty_midi.TimeSignature(4, 4, 0))
    
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    current_time = 0
    last_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_tensor = transform(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(frame_tensor)
            predicted = torch.sigmoid(output).cpu().numpy()

        key_indices = np.where(predicted >= 0.5)[1]
        for idx in key_indices:
            note_number = idx + 21 
            note_on = pretty_midi.Note(
                velocity=64, pitch=note_number, start=current_time, end=current_time + 1/frame_rate)
            piano.notes.append(note_on)
            note_off = pretty_midi.Note(
                velocity=0, pitch=note_number, start=current_time + 1/frame_rate, end=current_time + 2/frame_rate)
            piano.notes.append(note_off)

        last_time = current_time
        current_time += 1/frame_rate

    midi_file.instruments.append(piano)
    midi_file.write('output.mid')

    cap.release()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier(num_classes=91, encoder='resnet18', pretrained=True)
# TODO: path
model.load_state_dict(torch.load('...'))
model.to(device)
model.eval()

# TODO: path
video_path = '...'
process_video_to_midi(video_path, model, device)
