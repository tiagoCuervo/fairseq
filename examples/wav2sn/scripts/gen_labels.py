import sys
import os

def generate_train_itl(split, clarity_data_path):
    lengths = open(os.path.join(clarity_data_path, f"HA_outputs/train_sn/feats_wavlm/{split}.lengths")).readlines()
    lengths = [int(l) for l in lengths]

    mfcc_file = os.path.join(clarity_data_path, f"HA_outputs/train_sn/{split}.km")

    cluster_file_ref = os.path.join(clarity_data_path, f"HA_outputs/train_sn/mfcc/cls500_idx/{split}_signal_0_1.km")
    cluster_file_interferer = os.path.join(clarity_data_path, f"HA_outputs/train_sn/mfcc/cls500_idx/{split}_noise_0_1.km")

    with open(cluster_file_ref, "r") as f:
        ref_k_lines = f.readlines()
    with open(cluster_file_interferer, "r") as f:
        ifr_k_lines = f.readlines()

    with open(mfcc_file, 'w') as f:
        for ref_k, ifr_k in zip(ref_k_lines, ifr_k_lines):
            ref_k = ref_k.strip()
            ifr_k = ifr_k.strip()
            f.write(ref_k + ' ' + ifr_k + "\n")

    print(f"Generated {mfcc_file}")

if __name__ == "__main__":
    # import ptvsd
    # ptvsd.enable_attach(('0.0.0.0', 7310))
    # print("Attach debugger now")
    # ptvsd.wait_for_attach()

    if len(sys.argv) != 3:
        print("Usage: python script.py <split> <clarity_data_path>")
        sys.exit(1)

    split = sys.argv[1]
    clarity_data_path = sys.argv[2]
    generate_train_itl(split, clarity_data_path)
 