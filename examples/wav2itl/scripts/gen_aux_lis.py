import json
import sys

def generate_train_lis_file(X, split, clarity_data_path):
    input_tsv_path = f"{clarity_data_path}/HA_outputs/{'test' if split == 'test' else 'train'}.{X}/{split}.tsv"
    metadata_file_path = f"{clarity_data_path}/metadata/listeners.json"
    output_file_path = f"{clarity_data_path}/HA_outputs/{'test' if split == 'test' else 'train'}.{X}/{split}.lis"

    metadata = load_metadata(metadata_file_path)

    with open(input_tsv_path, 'r') as tsv_file, open(output_file_path, 'w') as output_file:
        for i, line in enumerate(tsv_file):
            if i == 0:
                continue
            file_path = line.split('\t')[0]
            listener_id = extract_listener_id(file_path)

            # audiogram_cfs = metadata[listener_id]['audiogram_cfs']
            audiogram_levels_l = metadata[listener_id]['audiogram_levels_l']
            audiogram_levels_r = metadata[listener_id]['audiogram_levels_r']

            # concatenated_array = audiogram_cfs + audiogram_levels_l + audiogram_levels_r
            concatenated_array = audiogram_levels_l + audiogram_levels_r
            concatenated_string = ' '.join(map(str, concatenated_array))
            output_file.write(concatenated_string + '\n')

    print(f"Generated {output_file_path}")

def load_metadata(metadata_file_path):
    with open(metadata_file_path, 'r') as metadata_file:
        metadata = json.load(metadata_file)
    return metadata

def extract_listener_id(file_path):
    listener_id = file_path.split('/')[1].split('_')[1]
    return listener_id

if __name__ == '__main__':
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
    generate_train_lis_file(X, split, clarity_data_path)
