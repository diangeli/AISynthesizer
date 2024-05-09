import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo
import cv2
import random
import os 
from tqdm import tqdm


def get_image_difference(background_path, new_image_path):

    # Read images
    background = cv2.imread(background_path)
    new_image = cv2.imread(new_image_path)

    # Resize images to the same size
    background = cv2.resize(background, (new_image.shape[1], new_image.shape[0]))

    # Convert images to grayscale
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    new_image_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    # Get absolute difference between the images
    difference = cv2.absdiff(background_gray, new_image_gray)

    return difference

def create_random_frame(background, frames, note_number):

    new_frame = background.copy()   
    notes = [] 
    used = []
    for i in range(note_number):
        while(True):
            note = int(random.random()*90)
            if note not in used: 
                break
            
        new_frame += frames[note] 
        notes.append(str(note))

    return new_frame, notes

def create_midi_from_notes(note_list):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(MetaMessage('set_tempo', tempo=bpm2tempo(120)))
    track.append(MetaMessage('time_signature', numerator=6, denominator=8))

    counter = 0
    for notes in note_list:
        for note in notes:
            note_on = Message('note_on', note=19+int(note), velocity=64, time=0)
            track.append(note_on)
        for note in notes:
            note_off = Message('note_off', note=19+int(note), velocity=64, time=int(1000/int(len(notes))))
            track.append(note_off)
        counter += 1

    return mid

def create_video_from_frames(frames, output_file:str):
    ### Create video
    # Get the shape of the frames
    height, width, _ = frames[0].shape

    # Define the output video file name
    output_file = output_file

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 30.0, (width, height))

    # Write each frame to the video file
    for frame in frames:
        out.write(frame)

    # Release the VideoWriter object
    out.release()

if __name__ == '__main__':

    # Open the video file
    video_capture = cv2.VideoCapture('all_notes.mid.mp4')

    # Check if the video file opened successfully
    if not video_capture.isOpened():
        print("Error: Couldn't open the video file.")
        exit()

    # Iterate through frames
    frame_count = 0
    background = 0
    gen = False
    notes = {}
    note_count = 0
    diff_mean_prev = 10
    while True:
        # Read the next frame
        ret, frame = video_capture.read()

        # Check if the frame was read successfully
        if not ret:
            break

        diff = cv2.absdiff(background, frame)
        diff_mean = cv2.mean(diff)[0]

        if diff_mean - diff_mean_prev > 0.1:
            notes[note_count] = frame
            note_count += 1
            cv2.imshow('Frame', diff)

        diff_mean_prev = diff_mean

        # Get background frame
        if frame_count == 0:
            background = frame
            frame_count += 1
            gen = True
            continue
            

        # Wait for the 'q' key to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        frame_count += 1
        

    # Release the video capture object and close any open windows
    video_capture.release()
    cv2.destroyAllWindows()

    # print("Total frames:", frame_count)

    # Create output folder if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')

    video_length = 5
    for video in tqdm(range(10)):
        video_frames = []
        video_notes = []
        for i in range(video_length):
            frame, played_notes = create_random_frame(background, notes, 1+int(random.random()*5))
            video_notes.append(played_notes)
            for i in range(30):
                video_frames.append(frame)

        ### Create video
        create_video_from_frames(video_frames, os.path.join('output', f'{video}.mp4'))

        ### Create midi
        create_midi_from_notes(video_notes).save(os.path.join('output', f'{video}.mid'))
