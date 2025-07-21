import sys

def process_file(filename):
    sequences = {}
    
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2 and not parts[0].startswith("#"):
                key, value = parts
                if key not in sequences:
                    sequences[key] = []
                sequences[key].append(value)
    
    for key, values in sequences.items():
        print(f">{key}")
        print("".join(values))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_file>")
        sys.exit(1)
    
    process_file(sys.argv[1])
