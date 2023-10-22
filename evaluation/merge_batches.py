import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-paths', type=str, nargs='+', help='List of input file paths')
    parser.add_argument('--output-path', type=str, help='Output file path')

    args = parser.parse_args()

    outputs = list()
    
    for in_path in args.input_paths:
        with open(in_path, 'r') as infile:
            outputs.extend(json.load(infile))

    with open(args.output_path, 'w') as outfile: 
        json.dump(outputs, outfile, indent=4, sort_keys=True)

if __name__ == "__main__":
    main()
