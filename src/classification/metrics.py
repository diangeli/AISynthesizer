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

ground_truth_dir = 'midis'
predicted_dir = 'classification_midis/ResNet18'
avg_precision, avg_recall, avg_f1 = evaluate_all_midis(ground_truth_dir, predicted_dir)
print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1 Score: {avg_f1}")

ground_truth_dir = 'midis'
predicted_dir = 'classification_midis/ResNet18_pt'
avg_precision, avg_recall, avg_f1 = evaluate_all_midis(ground_truth_dir, predicted_dir)
print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1 Score: {avg_f1}")

ground_truth_dir = 'midis'
predicted_dir = 'classification_midis/ResNet50'
avg_precision, avg_recall, avg_f1 = evaluate_all_midis(ground_truth_dir, predicted_dir)
print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1 Score: {avg_f1}")


ground_truth_dir = 'midis'
predicted_dir = 'classification_midis/ResNet50_pt'
avg_precision, avg_recall, avg_f1 = evaluate_all_midis(ground_truth_dir, predicted_dir)
print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1 Score: {avg_f1}")


ground_truth_dir = 'midis'
predicted_dir = 'classification_midis/ViT16'
avg_precision, avg_recall, avg_f1 = evaluate_all_midis(ground_truth_dir, predicted_dir)
print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1 Score: {avg_f1}")


ground_truth_dir = 'midis'
predicted_dir = 'classification_midis/ViT16_pt'
avg_precision, avg_recall, avg_f1 = evaluate_all_midis(ground_truth_dir, predicted_dir)
print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1 Score: {avg_f1}")


ground_truth_dir = 'midis'
predicted_dir = 'classification_midis/ViT32'
avg_precision, avg_recall, avg_f1 = evaluate_all_midis(ground_truth_dir, predicted_dir)
print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1 Score: {avg_f1}")


ground_truth_dir = 'midis'
predicted_dir = 'classification_midis/ViT32_pt'
avg_precision, avg_recall, avg_f1 = evaluate_all_midis(ground_truth_dir, predicted_dir)
print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1 Score: {avg_f1}")
