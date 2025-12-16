#!/bin/bash

# Check if path was provided
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_repository>"
  exit 1
fi

path="$1"

# Loop through all .fasta files in the given directory
for fasta_file in "$path"/*.fasta; do
  # Extract the label from the filename (remove path and extension)
  filename=$(basename "$fasta_file")
  label="${filename%.fasta}"

  echo "Running adabmDCA on label: $label"
  adabmDCA profmark -t1 0.4 -t2 1.0 -t3 0.7 --bestof 1 "$path/$label" "$fasta_file"
done
