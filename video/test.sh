#!/usr/bin/env bash

currentFolder=$(basename "$PWD")
outputFile="output_${currentFolder}.txt"
gtFile="${currentFolder}_GT.mp4"

echo "Current folder: $currentFolder"
echo "Output file: $outputFile"
echo "GT file: $gtFile"

conda activate fvvdp

: > "$outputFile"

# loop over all mp4s except the GT file
for mp4 in *.mp4; do
  if [ "$mp4" != "$gtFile" ]; then
    echo "Processing: $mp4"
    echo "Processing: $mp4" >> "$outputFile"
    result=$(fvvdp --test "$mp4" --ref "$gtFile" --display standard_fhd 2>&1)
    echo "$result" >> "$outputFile"
  fi
done

read -n1 -r -p $'Press Q (or any key) to exit...\n'
