import os
import pretty_midi
import mir_eval
import numpy as np

def load_midi_notes(midi_file):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    notes = [(note.pitch, note.start, note.end) for note in midi_data.instruments[0].notes if (note.pitch != 107 or note.pitch != 21)]
    return notes

def compare_midi_files(pred_midi_file, true_midi_file):
    pred_notes = load_midi_notes(pred_midi_file)
    true_notes = load_midi_notes(true_midi_file)

    pred_intervals = np.array([[start, end] for _, start, end in pred_notes])
    pred_pitches = np.array([pitch for pitch, _, _ in pred_notes])
    true_intervals = np.array([[start, end] for _, start, end in true_notes])
    true_pitches = np.array([pitch for pitch, _, _ in true_notes])

    if true_intervals.size > 0 and pred_intervals.size > 0:
        precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            true_intervals, true_pitches, pred_intervals, pred_pitches, onset_tolerance=0.05)
    else:
        precision, recall, f1 = 0, 0, 0

    return precision, recall, f1

def evaluate_all_midis(ground_truth_dir, predicted_dir):
    scores = {'precision': [], 'recall': [], 'f1': []}

    for ground_truth_filename in os.listdir(ground_truth_dir):
        if ground_truth_filename.endswith('.mid'):
            ground_truth_path = os.path.join(ground_truth_dir, ground_truth_filename)
            predicted_path = os.path.join(predicted_dir, ground_truth_filename)

            precision, recall, f1 = compare_midi_files(predicted_path, ground_truth_path)
            scores['precision'].append(precision)
            scores['recall'].append(recall)
            scores['f1'].append(f1)

    avg_precision = np.mean(scores['precision']) if scores['precision'] else 0
    avg_recall = np.mean(scores['recall']) if scores['recall'] else 0
    avg_f1 = np.mean(scores['f1']) if scores['f1'] else 0

    return avg_precision, avg_recall, avg_f1

def midi_to_piano_roll(midi_file, fs=100):
    """Convert MIDI file to a Piano Roll array."""
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    piano_roll = midi_data.get_piano_roll(fs=fs)
    return (piano_roll > 0).astype(int)

def calculate_confusion_matrix(gt_piano_roll, pred_piano_roll):
    """Calculate the confusion matrix from two piano rolls."""
    TP = np.sum(np.logical_and(pred_piano_roll == 1, gt_piano_roll == 1))
    FP = np.sum(np.logical_and(pred_piano_roll == 1, gt_piano_roll == 0))
    TN = np.sum(np.logical_and(pred_piano_roll == 0, gt_piano_roll == 0))
    FN = np.sum(np.logical_and(pred_piano_roll == 0, gt_piano_roll == 1))
    return TP, FP, TN, FN

def confusion_matrix(ground_truth_dir, predicted_dir, fs=100):
    """Evaluate all MIDI files in given directories and compute a global confusion matrix."""
    global_TP, global_FP, global_TN, global_FN = 0, 0, 0, 0

    for ground_truth_filename in os.listdir(ground_truth_dir):
        if ground_truth_filename.endswith('.mid'):
            ground_truth_path = os.path.join(ground_truth_dir, ground_truth_filename)
            predicted_path = os.path.join(predicted_dir, ground_truth_filename)

            if os.path.exists(predicted_path):
                gt_piano_roll = midi_to_piano_roll(ground_truth_path, fs)
                pred_piano_roll = midi_to_piano_roll(predicted_path, fs)
                TP, FP, TN, FN = calculate_confusion_matrix(gt_piano_roll, pred_piano_roll)

                global_TP += TP
                global_FP += FP
                global_TN += TN
                global_FN += FN
            else:
                print(f"Predicted MIDI file not found for {ground_truth_filename}")

    return global_TP, global_FP, global_TN, global_FN


ground_truth_dir = 'midis'
predicted_dir = 'classification_midis/ResNet18'
avg_precision, avg_recall, avg_f1 = evaluate_all_midis(ground_truth_dir, predicted_dir)
print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1 Score: {avg_f1}")
global_TP, global_FP, global_TN, global_FN = confusion_matrix(ground_truth_dir, predicted_dir)
print(f"Global TP: {global_TP}, FP: {global_FP}, TN: {global_TN}, FN: {global_FN}")


predicted_dir = 'classification_midis/ResNet18_pt'
avg_precision, avg_recall, avg_f1 = evaluate_all_midis(ground_truth_dir, predicted_dir)
print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1 Score: {avg_f1}")
global_TP, global_FP, global_TN, global_FN = confusion_matrix(ground_truth_dir, predicted_dir)
print(f"Global TP: {global_TP}, FP: {global_FP}, TN: {global_TN}, FN: {global_FN}")


predicted_dir = 'classification_midis/ResNet50'
avg_precision, avg_recall, avg_f1 = evaluate_all_midis(ground_truth_dir, predicted_dir)
print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1 Score: {avg_f1}")
global_TP, global_FP, global_TN, global_FN = confusion_matrix(ground_truth_dir, predicted_dir)
print(f"Global TP: {global_TP}, FP: {global_FP}, TN: {global_TN}, FN: {global_FN}")


predicted_dir = 'classification_midis/ResNet50_pt'
avg_precision, avg_recall, avg_f1 = evaluate_all_midis(ground_truth_dir, predicted_dir)
print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1 Score: {avg_f1}")
global_TP, global_FP, global_TN, global_FN = confusion_matrix(ground_truth_dir, predicted_dir)
print(f"Global TP: {global_TP}, FP: {global_FP}, TN: {global_TN}, FN: {global_FN}")


predicted_dir = 'classification_midis/ViT16'
avg_precision, avg_recall, avg_f1 = evaluate_all_midis(ground_truth_dir, predicted_dir)
print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1 Score: {avg_f1}")
global_TP, global_FP, global_TN, global_FN = confusion_matrix(ground_truth_dir, predicted_dir)
print(f"Global TP: {global_TP}, FP: {global_FP}, TN: {global_TN}, FN: {global_FN}")


predicted_dir = 'classification_midis/ViT16_pt'
avg_precision, avg_recall, avg_f1 = evaluate_all_midis(ground_truth_dir, predicted_dir)
print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1 Score: {avg_f1}")
global_TP, global_FP, global_TN, global_FN = confusion_matrix(ground_truth_dir, predicted_dir)
print(f"Global TP: {global_TP}, FP: {global_FP}, TN: {global_TN}, FN: {global_FN}")


predicted_dir = 'classification_midis/ViT32'
avg_precision, avg_recall, avg_f1 = evaluate_all_midis(ground_truth_dir, predicted_dir)
print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1 Score: {avg_f1}")
global_TP, global_FP, global_TN, global_FN = confusion_matrix(ground_truth_dir, predicted_dir)
print(f"Global TP: {global_TP}, FP: {global_FP}, TN: {global_TN}, FN: {global_FN}")


predicted_dir = 'classification_midis/ViT32_pt'
avg_precision, avg_recall, avg_f1 = evaluate_all_midis(ground_truth_dir, predicted_dir)
print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1 Score: {avg_f1}")
global_TP, global_FP, global_TN, global_FN = confusion_matrix(ground_truth_dir, predicted_dir)
print(f"Global TP: {global_TP}, FP: {global_FP}, TN: {global_TN}, FN: {global_FN}")
