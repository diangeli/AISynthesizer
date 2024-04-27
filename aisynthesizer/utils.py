import mido
import numpy as np
import pretty_midi
import soundfile as sf
import torch
from omegaconf import DictConfig
from torchvision import transforms

from aisynthesizer.models.vivit import ViT


class Utils:
    def __init__(self, config: DictConfig):
        self.config = config
        self.transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),  # Resize to the input size expected by the model
                transforms.ToTensor(),  # Convert images to PyTorch tensors
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        if config.data.enable_transforms is False:
            self.train_transforms = self.test_transforms

    def get_transforms(self):
        return self.transforms

    def get_model(self):
        if self.config.model.name == "vivit":
            return ViT(
                image_size=(224, 224),  # Height and width of input frames
                num_frames=self.config.model.num_frames,  # Total number of frames in each video
                num_classes=88,  # For example, 88 keys on the piano
                dim=1024,  # Dimensionality of the token/patch embeddings
                depth=6,  # Number of transformer blocks (depth)
                heads=8,  # Number of attention heads
                mlp_dim=2048,  # Dimensionality of the feedforward layer
                pool="cls",  # Pooling method ('cls' for class token, 'mean' for mean pooling)
                channels=3,  # Number of channels in the video frames (RGB, so 3)
                dim_head=64,  # Dimensionality of each attention head
                dropout=0.1,  # Dropout rate
                emb_dropout=0.1,  # Embedding dropout rate
            )
        else:
            raise NotImplementedError(f"{self.config.model.name} model not yet supported!")

    def get_optimizer(self, model):
        if self.config.training.optimizer == "adam":
            return torch.optim.Adam(model.parameters(), lr=self.config.training.lr)
        else:
            raise NotImplementedError(f"{self.config.training.optimizer} optimizer not yet supported!")
        
    def process_roll(self, smodel_outputs):
        threshold = self.config.model.threshold
        print(smodel_outputs)
        smodel_outputs = smodel_outputs >= threshold
        print(smodel_outputs)
        # compute onsets and offsets
        onset = np.zeros(smodel_outputs.shape)
        offset = np.zeros(smodel_outputs.shape)
        for j in range(smodel_outputs.shape[0]):
            if j != 0:
                onset[j][np.setdiff1d(smodel_outputs[j].nonzero(),
                                        smodel_outputs[j - 1].nonzero())] = 1
                offset[j][np.setdiff1d(smodel_outputs[j - 1].nonzero(),
                                        smodel_outputs[j].nonzero())] = -1
            else:
                onset[j][smodel_outputs[j].nonzero()] = 1
        onset += offset
        
        print("The onset has shape:", onset.shape)
        onset = onset.T
        notes = {}
        for i in range(onset.shape[0]):
            tmp = onset[i]
            start = np.where(tmp == 1)[0]
            end = np.where(tmp == -1)[0]
            if len(start) != len(end):
                end = np.append(end, tmp.shape)
            merged_list = [(start[i], end[i]) for i in range(0, len(start))]
            # 21 is the lowest piano key in the Midi note number (Midi has 128 notes)
            notes[21 + i] = merged_list
        return notes

    def generate_midi(notes, output_file):
        instrument = 'Acoustic Grand Piano'
        pm = pretty_midi.PrettyMIDI(initial_tempo=80)
        piano_program = pretty_midi.instrument_name_to_program(instrument) #Acoustic Grand Piano
        piano = pretty_midi.Instrument(program=piano_program)
        for key in list(notes.keys()):
            values = notes[key]
            for i in range(len(values)):
                start, end = values[i]
                note = pretty_midi.Note(velocity=100, pitch=key, start=start * 0.04, end=end * 0.04)
                piano.notes.append(note)
        pm.instruments.append(piano)
        pm.write(output_file)
        wav = pm.fluidsynth(fs=16000)
        out_file = output_file.replace(".mid", ".wav")
        sf.write(out_file, wav, samplerate=16000)
        return wav

    def read_midi(file_path):
        # Load a MIDI file
        midi_file = mido.MidiFile(file_path)
        
        print(f"Reading MIDI file: {file_path}")
        print("Tracks:", len(midi_file.tracks))
        
        for track_idx, track in enumerate(midi_file.tracks):
            print(f"Track {track_idx}: {track.name if hasattr(track, 'name') else 'Unnamed'}")
            for msg in track:
                print(msg)
                if msg.type == 'note_on' and msg.velocity > 0:
                    print(f"Note On - Channel: {msg.channel}, Note: {msg.note}, Velocity: {msg.velocity}, Time: {msg.time}")
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    print(f"Note Off - Channel: {msg.channel}, Note: {msg.note}, Velocity: {msg.velocity}, Time: {msg.time}")
                elif msg.type == 'program_change':
                    print(f"Program Change - Channel: {msg.channel}, Program: {msg.program}, Time: {msg.time}")

    # def get_loss_function(self):
    #     if self.config.model.loss == "ce+":
    #         return nn.CrossEntropyLoss()
    #     elif self.config.model.loss == "dice":
    #         return DiceLoss('multiclass')
    #     elif self.config.model.loss == "focal":
    #         return FocalLoss('multiclass')
    #     else:
    #         return TverskyLoss('multiclass')

    def create_models_name(self, epoch: int):
        return f"{self.config.model.name}_{self.config.training.optimizer}_{self.config.training.lr}_epoch_{epoch}_out_of_{self.config.training.epochs}_frames_{self.config.model.num_frames}"
