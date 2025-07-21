import sys
import sys

def process_fasta(filename):
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                print(line, end="\n")
            else:
                filtered_line = "".join([char for char in line if char == "-" or char.isupper()])
                print(filtered_line, end="\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python removeInserts.py <input_file>")
        sys.exit(1)
    
    process_fasta(sys.argv[1])
    