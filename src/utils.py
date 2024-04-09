import mido
from mido import MidiFile, MidiTrack, Message
import numpy as np

def create_midi(model_outputs, epoch, batch_idx, video_idx, threshold=0.5):
    """
    Converts model output probabilities to a MIDI file.
    
    Args:
        model_outputs (np.ndarray): Numpy array of shape (num_frames, 88) containing note probabilities.
        epoch, batch_idx, video_idx (int): Identifiers for tracking and naming.
        threshold (float): Probability threshold for determining if a note is on.
    """
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    ticks_per_frame = 480  # This value might need to be adjusted based on the MIDI and the actual video frame rate
    track.append(Message('program_change', program=0, time=0))  # Setting the MIDI instrument to Acoustic Grand Piano

    for frame_idx, frame_probs in enumerate(model_outputs):
        # Ensure frame_probs is iterable
        if not isinstance(frame_probs, np.ndarray):
            raise ValueError(f"Expected frame_probs to be an np.ndarray, got {type(frame_probs)}")
        for note_idx, prob in enumerate(frame_probs):
            note_number = note_idx + 21  # MIDI notes for a piano start at 21
            if prob > threshold:
                # Note on
                track.append(Message('note_on', note=note_number, velocity=64, time=ticks_per_frame * frame_idx))
                # Note off
                track.append(Message('note_off', note=note_number, velocity=64, time=ticks_per_frame * (frame_idx + 1)))

    # Save the MIDI file
    midi_filename = f"midi_epoch{epoch+1}_batch{batch_idx+1}_video{video_idx+1}.mid"
    mid.save(midi_filename)
    print(f"MIDI file saved as {midi_filename}")
