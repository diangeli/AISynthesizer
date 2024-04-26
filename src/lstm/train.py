import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import time
import numpy as np
from sklearn.utils import shuffle
import processing as Processing
from model import LSTMModel
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

def sample_index(preds, temperature=1.0):
    preds = np.clip(preds, 1e-10, None)  
    
    try:
        preds = np.log(preds) / temperature  
    except RuntimeWarning:
        raise ValueError("Invalid operation during `np.log(preds)`, possibly due to zeros or negative values.")
    
    exp_preds = np.exp(preds)
    sum_exp_preds = np.sum(exp_preds)

    if not (0.999 <= sum_exp_preds <= 1.001):
        raise ValueError("Sum of probabilities is outside the expected range (0.999 to 1.001).")

    normalized_preds = exp_preds / sum_exp_preds

    if np.any(np.isnan(normalized_preds)) or np.any(normalized_preds < 0):
        raise ValueError("Probability values contain NaNs or negatives after normalization.")

    probs = np.random.multinomial(1, normalized_preds, 1)
    return np.argmax(probs)  


# training data from all MIDI files
def prepare_training_data(training_inputs, maxlen, notes):
    sentences = []
    next_chars = []
    note_indices = {c: i for i, c in enumerate(notes)}

    for midi_file in training_inputs:
        note_string = Processing.midi_file_to_string(midi_file)
        if len(note_string) >= maxlen:
            for i in range(len(note_string) - maxlen):
                sentences.append(note_string[i : i + maxlen])
                next_chars.append(note_string[i + maxlen])

    x = torch.zeros((len(sentences), maxlen, len(notes)), dtype=torch.float32)
    y = torch.zeros((len(sentences), len(notes)), dtype=torch.float32)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            if char in note_indices:
                x[i, t, note_indices[char]] = 1.0
        if next_chars[i] in note_indices:
            y[i, note_indices[next_chars[i]]] = 1.0
    
    return x, y

def train_network(model, optimizer, loss_fn, x, y, epochs=20, batch_size=128, patience=5):
    x, y = shuffle(x, y)
    
    split_idx = int(0.8 * len(x))
    x_train, y_train = x[:split_idx], y[:split_idx]
    x_val, y_val = x[split_idx:], y[split_idx:]

    best_val_loss = float('inf')
    patience_counter = patience

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for start in range(0, len(x_train), batch_size):
            x_batch = x_train[start:start + batch_size]
            y_batch = y_train[start:start + batch_size]

            optimizer.zero_grad()  
            output = model(x_batch)
            loss = loss_fn(output, y_batch)  
            loss.backward() 
            clip_grad_norm_(model.parameters(), 5.0)  # Limit gradient norm
            optimizer.step() 

            total_train_loss += loss.item()

        # Evaluate model
        model.eval()
        with torch.no_grad():
            val_output = model(x_val)
            val_loss = loss_fn(val_output, y_val)  
            scheduler.step(val_loss)  

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {total_train_loss / len(x_train)}, Validation Loss: {val_loss.item()}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss 
            patience_counter = patience
        else:
            patience_counter -= 1  

        if patience_counter <= 0:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    if not os.path.isdir("checkpoints"):
        os.mkdir("checkpoints")
    
    torch.save(model.state_dict(), "checkpoints/final_model_weights.pt")
    print("Model saved.")

def generate_text(model, seed_string, note_indices, indices_note, maxlen, length=800, temperature=1.0):
    if len(seed_string.split()) < maxlen:
        raise ValueError("Seed string must have a minimum length equal to 'maxlen'.")

    sentence = " ".join(seed_string.split()[:maxlen])
    generated = sentence  
    model.eval()

    with torch.no_grad():  # for inference
        for _ in range(length):
            x_pred = torch.zeros((1, maxlen, len(note_indices)), dtype=torch.float32)

            for t, char in enumerate(sentence.split(" ")):  # handling of space-separated values
                if char in note_indices:
                    x_pred[0, t, note_indices[char]] = 1.0
                else:
                    raise ValueError(f"Unexpected character '{char}' in the sentence.")

            preds = model(x_pred)
            preds = F.softmax(preds, dim=1)

            preds = preds.detach().numpy()[0]

            next_index = sample_index(preds, temperature)

            if next_index not in indices_note:
                raise ValueError(f"Generated index '{next_index}' is invalid.")

            next_char = indices_note[next_index]
            sentence = " ".join(sentence.split(" ")[1:] + [next_char])  
            generated += f" {next_char}"

    return generated

def main():
    if not os.path.isdir("midis"):
        print("Error: The 'midis' folder does not exist.")
        return

    notes = list(map(str, range(128)))  
    maxlen = 5

    training_inputs = ["midis/" + midi for midi in os.listdir("midis") if midi.endswith(('.mid', '.midi'))]
    # training_inputs = training_inputs[:200]
    if not training_inputs:
        print("Error: No MIDI files found in the 'midis' folder.")
        return

    x, y = prepare_training_data(training_inputs, maxlen, notes)

    model = LSTMModel(input_size=len(notes), hidden_size1=128, hidden_size2=128, output_size=len(notes))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss() 

    train_network(model, optimizer, loss_fn, x, y, epochs=20, patience=5)

    random_midi = Processing.get_random_midi("midis")
    if not random_midi:
        print("Error: Could not find a random MIDI file.")
        return

    generated_string = Processing.midi_file_to_string(random_midi)
    if not generated_string or len(generated_string) < maxlen:
        print("Error: 'generated_string' is empty or shorter than 'maxlen'.")
        return

    output_name = f"output_test"
    generated_text = generate_text(model, generated_string, {str(i): i for i in range(128)}, {i: str(i) for i in range(128)}, maxlen, 20)

    Processing.string_to_wav(generated_text, output_name)
    print("Training complete. WAV file generated from a random MIDI.")

if __name__ == "__main__":
    main()
