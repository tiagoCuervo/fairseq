import json
import sys
import os

def generate_train_itl(X, split, clarity_data_path):
    tsv_file = os.path.join(clarity_data_path, f"HA_outputs/{split}.{X}/{split}.tsv")

    with open(tsv_file, 'r') as tsv:
        lines = tsv.readlines()

    itl_file = os.path.join(clarity_data_path, f"HA_outputs/{split}.{X}/{split}.itl")

    with open(itl_file, 'w') as itl:
        for line in lines[1:]:
            path = line.strip().split('\t')[0]
            y_value = path[3]
            signal_id = path.split('/')[-1].split('.')[0]
            metadata_file = os.path.join(clarity_data_path, f"metadata/CEC{y_value}.{split}.{X}.json")

            with open(metadata_file, 'r') as metadata:
                metadata = json.load(metadata)

            for item in metadata:
                if item['signal'] == signal_id:
                    correctness = float(item['correctness']) / 100
                    itl.write(f"{correctness}\n")
                    break

    print(f"Generated {itl_file}")

if __name__ == "__main__":
    # import ptvsd
    # ptvsd.enable_attach(('0.0.0.0', 7310))
    # print("Attach debugger now")
    # ptvsd.wait_for_attach()
    
    if len(sys.argv) != 4:
        print("Usage: python script.py <X> <split> <clarity_data_path>")
        sys.exit(1)

    X = sys.argv[1]
    split = sys.argv[2]
    clarity_data_path = sys.argv[3]
    generate_train_itl(X, split, clarity_data_path)
 