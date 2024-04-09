import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
import cv2
import os
import mido  # Ensure 'mido' is installed for MIDI file processing

class VideoDataset(Dataset):
    def __init__(self, video_dir, midi_dir, transforms=None):
        self.video_dir = video_dir
        self.midi_dir = midi_dir
        self.transforms = transforms
        self.videos = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')])
        self.midi_files = sorted([os.path.join(midi_dir, f) for f in os.listdir(midi_dir) if f.endswith('.mid')])

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        # Now correctly passing the 'transforms' argument as well
        video_tensor = self.load_video_frames_as_tensor(video_path, self.transforms)

        midi_path = self.midi_files[idx]
        labels = self.midi_to_label_vector(midi_path, num_frames=125)  # Assuming 'midi_to_label_vector' now correctly accepts a file path and num_frames

        return {'video': video_tensor, 'midi_labels': labels}

    def load_video_frames_as_tensor(self, video_path, transforms):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frame = ToTensor()(frame)  # Convert to tensor
            if transforms is not None:
                frame = transforms(frame)
            frames.append(frame)
        cap.release()
        frames_tensor = torch.stack(frames)  # Stack frames into a single tensor
        return frames_tensor
    
    @staticmethod

    def midi_to_label_vector(midi_file_path, num_frames=125):
        midi_data = mido.MidiFile(midi_file_path)
        labels = np.zeros((num_frames, 88), dtype=np.float32)
        total_ticks = sum(msg.time for track in midi_data.tracks for msg in track)
        ticks_per_frame = total_ticks / num_frames
        current_frame = 0
        accumulated_ticks = 0

        for track in midi_data.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    note_index = msg.note - 21
                    if 0 <= note_index < 88:
                        accumulated_ticks += msg.time
                        while accumulated_ticks >= ticks_per_frame and current_frame < num_frames - 1:
                            accumulated_ticks -= ticks_per_frame
                            current_frame += 1
                        labels[current_frame, note_index] = 1

        return labels

