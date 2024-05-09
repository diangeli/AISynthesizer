import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
import cv2
import os
import mido 
import itertools as it

class VideoDataset(Dataset):
    def __init__(self, video_dir, midi_dir=None, num_frames = 125, transforms=None, mode='train', limit=None):
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
        self.num_frames = num_frames
        self.videos = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')])
        if midi_dir:
            self.midi_files = sorted([os.path.join(midi_dir, f) for f in os.listdir(midi_dir) if f.endswith('.mid')])
        else:
            self.midi_files = [None] * len(self.videos)  # Placeholder when no MIDI files are present
        self.target_videos = None
        self.target_midis = None
        self.load_data()

        # Apply limit if specified
        # if limit is not None:
        #     self.videos = self.videos[:limit]
        #     if midi_dir:
        #         self.midi_files = self.midi_files[:limit]
        #     else:
        #         self.midi_files = [None] * len(self.videos)

    def __len__(self):
        return len(self.target_videos)

    def __getitem__(self, idx):
        video_path, target_frame_num, target_frames = self.target_videos[idx]
        video_tensor = self.load_video_frames_as_tensor(video_path, target_frames, self.transforms)

        if self.mode == 'train' and self.target_midis[idx] is not None:
            midi_path, target_frame_num, total_frames = self.target_midis[idx]
            labels = self.midi_to_label_vector(midi_path, target_frame_num, total_frames)
            return {'video': video_tensor, 'midi_labels': labels}
        else:
            return {'video': video_tensor}

    def load_video_frames_as_tensor(self, video_path, target_frames, transforms):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        for frame_num in target_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            success, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frame = ToTensor()(frame)  # Convert to tensor
            if transforms:
                frame = transforms(frame)
            frames.append(frame)
            success, frame = cap.read()

        while len(frames) < self.num_frames: 
            print("Something is wrongg!!!!")
            frames.append(torch.zeros_like(frames[0])) 

        frames_tensor = torch.stack(frames)
        cap.release()
        return frames_tensor       
    
    @staticmethod
    def midi_to_label_vector(midi_file_path, target_frame, total_frames):
        midi_data = mido.MidiFile(midi_file_path)
        labels = np.zeros((total_frames, 88), dtype=np.float32)
        total_ticks = sum(msg.time for track in midi_data.tracks for msg in track)
        ticks_per_frame = total_ticks / total_frames
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
                    if current_frame >= total_frames:
                        break

                if msg.type == 'note_on':
                    note_index = msg.note - 21
                    if 0 <= note_index < 88:
                        note_active[note_index] = msg.velocity > 0
                elif msg.type == 'note_off':
                    note_index = msg.note - 21
                    if 0 <= note_index < 88:
                        note_active[note_index] = False
        return labels[target_frame]
    
    def load_data(self):
        self.target_videos = []
        self.target_midis = []
        i = 0
        for video_path, midi_path in zip(self.videos, self.midi_files):
            if i >= 30:
                break
            video = cv2.VideoCapture(str(video_path))
            total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
            half = int(self.num_frames / 2)
            for frame_num in range(int(total_frames)):
                target_frames = list(range(frame_num-half,frame_num+half+1))
                target_frames = [frame if frame >= 0 else 0 for frame in target_frames ]
                target_frames = [frame if frame < total_frames else total_frames-1 for frame in target_frames ]
                self.target_videos.append((video_path,frame_num,target_frames))
                self.target_midis.append((midi_path, frame_num, int(total_frames)))
            i+=1
                

