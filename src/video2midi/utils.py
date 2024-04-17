import mido
from mido import MidiFile, MidiTrack, Message
import numpy as np
from typing import Any, Callable
import jax.numpy as jnp


def create_midi(model_outputs, output_file, threshold=0.5, min_duration=480, max_notes_per_frame=1):
    """
    Converts model output probabilities to a MIDI file, adding more realistic durations and limiting polyphony.
    
    Args:
        model_outputs (np.ndarray): Numpy array of shape (num_frames, 88) containing note probabilities.
        output_file (str): Path where the MIDI file will be saved.
        threshold (float): Probability threshold for determining if a note is on.
        min_duration (int): Minimum duration of a note in MIDI ticks.
        max_notes_per_frame (int): Maximum number of notes allowed per frame.
    """
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(Message('program_change', program=0, time=0))  # Acoustic Grand Piano

    current_notes = []  # Tracks notes that are currently playing

    for frame_idx, frame_probs in enumerate(model_outputs):
        frame_events = []

        # Check for notes above the threshold
        for note_idx, prob in enumerate(frame_probs):
            if prob > threshold:
                note_number = note_idx + 21
                frame_events.append((note_number, prob))

        # Sort by probability and enforce maximum notes per frame
        frame_events.sort(key=lambda x: x[1], reverse=True)  # Sort by probability, descending
        frame_events = frame_events[:max_notes_per_frame]

        # Turn off all current notes not in the new set of frame events
        new_notes_set = set(note for note, _ in frame_events)
        for note_number in current_notes:
            if note_number not in new_notes_set:
                track.append(Message('note_off', note=note_number, velocity=64, time=0))

        # Update currently playing notes
        current_notes = list(new_notes_set)

        # Trigger new notes
        for note_number, _ in frame_events:
            track.append(Message('note_on', note=note_number, velocity=64, time=min_duration if frame_idx > 0 else 0))

    # Turn off any remaining notes
    for note_number in current_notes:
        track.append(Message('note_off', note=note_number, velocity=64, time=min_duration))

    mid.save(output_file)
    print(f"MIDI file saved as {output_file}")


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

# if __name__ == "__main__":
#     # Specify the path to your MIDI file
#     midi_path = "results_midis/midi_output_0.mid"
#     read_midi(midi_path)
#     midi_path = "midis/track_0.mid"
#     read_midi(midi_path)

