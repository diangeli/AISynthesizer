import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
import cv2
import os
import mido  

class VideoDataset(Dataset):
    def __init__(self, video_dir, midi_dir=None, transforms=None, mode='train', limit=None):
        """
        Initializes the dataset loader.
        
        Args:
            video_dir (str): Directory containing videos.
            midi_dir (str, optional): Directory containing MIDI files, if available.
            transforms (callable, optional): Transformations to apply to video frames.
            mode (str): 'train' for training mode with labels, 'eval' for evaluation without labels.
            limit (int, optional): Limit the number of videos and MIDI files to load.
        """
        self.video_dir = video_dir
        self.midi_dir = midi_dir
        self.transforms = transforms
        self.mode = mode
        self.videos = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')])
        if midi_dir:
            self.midi_files = sorted([os.path.join(midi_dir, f) for f in os.listdir(midi_dir) if f.endswith('.mid')])
        else:
            self.midi_files = [None] * len(self.videos)  # Placeholder when no MIDI files are present

        # Apply limit if specified
        if limit is not None:
            self.videos = self.videos[:limit]
            if midi_dir:
                self.midi_files = self.midi_files[:limit]
            else:
                self.midi_files = [None] * len(self.videos)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        video_tensor = self.load_video_frames_as_tensor(video_path, self.transforms)

        if self.mode == 'train' and self.midi_files[idx] is not None:
            midi_path = self.midi_files[idx]
            labels = self.midi_to_label_vector(midi_path, num_frames=125)
            return {'video': video_tensor, 'midi_labels': labels}
        else:
            return {'video': video_tensor}

    def load_video_frames_as_tensor(self, video_path, transforms):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        success, frame = cap.read()
        while success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frame = ToTensor()(frame)  # Convert to tensor
            if transforms:
                frame = transforms(frame)
            frames.append(frame)
            success, frame = cap.read()

        while len(frames) < 125: 
            frames.append(torch.zeros_like(frames[0])) 

        frames_tensor = torch.stack(frames)
        cap.release()
        return frames_tensor        
    
    @staticmethod
    def midi_to_label_vector(midi_file_path, num_frames=125):
        midi_data = mido.MidiFile(midi_file_path)
        labels = np.zeros((num_frames, 88), dtype=np.float32)
        total_ticks = sum(msg.time for track in midi_data.tracks for msg in track)
        ticks_per_frame = total_ticks / num_frames
        current_frame = 0
        accumulated_ticks = 0

        note_active = np.zeros(88, dtype=bool)  # Track whether a note is active

        for track in midi_data.tracks:
            for msg in track:
                accumulated_ticks += msg.time
                while accumulated_ticks >= ticks_per_frame:
                    labels[current_frame] = note_active.astype(float)
                    accumulated_ticks -= ticks_per_frame
                    current_frame += 1
                    if current_frame >= num_frames:
                        break

                if msg.type == 'note_on':
                    note_index = msg.note - 21
                    if 0 <= note_index < 88:
                        note_active[note_index] = msg.velocity > 0
                elif msg.type == 'note_off':
                    note_index = msg.note - 21
                    if 0 <= note_index < 88:
                        note_active[note_index] = False

        return labels

