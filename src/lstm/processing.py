import random
import os
import py_midicsv
import pretty_midi
from scipy.io.wavfile import write
import numpy as np

def midi_file_to_string(filename):
    csv_string = py_midicsv.midi_to_csv(filename)
    print(csv_string)
    note_on = "Note_on_c"
    note_off = "Note_off_c"
    eof = "End_of_file"

    final_string = ""

    for line in csv_string:
        note_info = line.split(",")  
        event_type = note_info[2].strip()  
        
        if event_type == "Note_on_c":
            note_value = int(note_info[4].strip())  
            final_string += f"{note_value} "  
    
    final_string = final_string.strip()  
    print(final_string)

    print(f"Generated final string from MIDI: '{final_string}'")
    
    return final_string


def string_to_notes(s):
    sanitized_string = ' '.join(s.split()).strip() 

    print(f"Sanitized string for conversion: '{sanitized_string}'")
    print(len(sanitized_string))

    note_tokens = sanitized_string.split(" ")

    if "" in note_tokens:
        raise ValueError("Sanitized string contains empty tokens or invalid characters.")

    list_note_strings = []
    list_note_strings.append("0, 0, Header, 1, 2, 384\n")
    list_note_strings.append("1, 0, Start_track\n")
    list_note_strings.append("1, 0, Program_c, 0, 0\n")

    time = 0
    for note in note_tokens:
        if note.isnumeric():
            note_value = int(note)
            if 0 <= note_value <= 127:  # Ensure valid MIDI note range
                note_str = f"1, {time}, Note_on_c, 0, {note_value}, 64\n"
                list_note_strings.append(note_str)
                time += 480  
                note_off_str = f"1, {time}, Note_off_c, 0, {note_value}, 0\n"
                list_note_strings.append(note_off_str)
                time += 480  
        else:
            raise ValueError(f"Unexpected note format: '{note}'")

    list_note_strings.append(f"1, {time}, End_track\n")
    list_note_strings.append("0, 0, End_of_file\n")

    return list_note_strings


def clean_midi_notes(midi_note_strings):
    max_zero_sequence_length = 3 
    cleaned_notes = []
    for note in midi_note_strings:
        corrected_note = note
        invalid_sequence = "0" * (max_zero_sequence_length + 1)

        while invalid_sequence in corrected_note:
            corrected_note = corrected_note.replace(invalid_sequence, "0")

        cleaned_notes.append(corrected_note)
    
    return cleaned_notes


def string_to_wav(s, output_name):
    notes = string_to_notes(s)

    notes = clean_midi_notes(notes)

    csv_filename = f"{output_name}.csv"
    with open(csv_filename, "w") as csv_file:
        csv_file.writelines(notes)

    midi_filename = f"{output_name}.mid"
    midi_obj = py_midicsv.csv_to_midi(csv_filename)

    with open(midi_filename, "wb") as midi_file:
        midi_writer = py_midicsv.FileWriter(midi_file)
        midi_writer.write(midi_obj)

    midi_to_wav(midi_filename, output_name)


# Converts MIDI to WAV format
def midi_to_wav(midi_filename, outputname):
    midi_format = pretty_midi.PrettyMIDI(midi_filename)
    audio_data = midi_format.synthesize()
    write(f"{outputname}.wav", 44100, audio_data)


# Gets a random MIDI file from the specified folder
def get_random_midi(folder_path):
    midi_files = [f for f in os.listdir(folder_path) if f.endswith(".mid") or f.endswith(".midi")]
    
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found in the specified folder: {folder_path}")

    return os.path.join(folder_path, random.choice(midi_files))
