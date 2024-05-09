#!/bin/bash

for file in output/*.wav; do
    number=$(basename "$file" .wav)
    output_file="output/${number}_combined.mp4"
    ffmpeg -i "output/$number.mp4" -i "$file" -vcodec copy "$output_file"
done

