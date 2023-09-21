# import json
import torch
import os
import numpy as np
from WavLM import WavLM, WavLMConfig
import skimage.measure
import argparse
import soundfile as sf
from shutil import copyfile
from npy_append_array import NpyAppendArray
import tqdm


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data", help="Path with .tsv files pointing to the audio data")
    parser.add_argument("--split", help="Which split", required=True)
    parser.add_argument("--save-dir", help="Output path to store the features", required=True)
    parser.add_argument("--checkpoint", help="Path to the WavLM checkpoint", required=True)
    return parser


def get_iterator(args, mdl, cfg):
    with open(os.path.join(args.data, args.split) + ".tsv", "r") as fp:
        lines = fp.read().split("\n")
        root = lines.pop(0).strip()
        files = [os.path.join(root, line.split("\t")[0]) for line in lines if len(line) > 0]
        num = len(files)

        def iterate():
            for fname in files:
                # feats = reader.get_feats(fname)
                wav, sr = sf.read(fname)
                wav = torch.from_numpy(wav).float().cuda()
                if wav.dim() > 1:
                    feats = []
                    for d in range(wav.dim()):
                        m_in = wav[:, d].view(1, -1)
                        with torch.no_grad():   
                            if cfg.normalize:
                                m_in = torch.nn.functional.layer_norm(m_in , m_in.shape)
                            _, audio_rep = mdl.extract_features(m_in, output_layer=mdl.cfg.encoder_layers, ret_layer_results=True)[0]
                            audio_rep = [rep[0] for rep in audio_rep]
                            audio_rep = torch.concatenate(audio_rep, dim=1).transpose(1, 0)
                            audio_rep = audio_rep.cpu().numpy()
                        audio_rep = skimage.measure.block_reduce(audio_rep, (1, 20, 1), np.mean) # downsample x20
                        audio_rep = np.transpose(audio_rep, (1, 0, 2)) # [time, heads, width]
                        feats.append(audio_rep)
                    yield np.concatenate([np.expand_dims(feats[0], axis=1), 
                                          np.expand_dims(feats[1], axis=1)], axis=1)
                else:
                    wav = wav.view(1, -1)
                    with torch.no_grad():
                        if cfg.normalize:
                            wav = torch.nn.functional.layer_norm(wav , wav.shape)
                        _, audio_rep = mdl.extract_features(wav, output_layer=mdl.cfg.encoder_layers, ret_layer_results=True)[0]
                        audio_rep = [rep[0] for rep in audio_rep]
                        audio_rep = torch.concatenate(audio_rep, dim=1).transpose(1, 0)
                        audio_rep = audio_rep.cpu().numpy()
                    audio_rep = skimage.measure.block_reduce(audio_rep, (1, 20, 1), np.mean) # downsample x20
                    audio_rep = np.transpose(audio_rep, (1, 0, 2))
                    yield audio_rep
    return iterate, num


def main():
    parser = get_parser()
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    def create_files(dest):
        copyfile(os.path.join(args.data, args.split) + ".tsv", dest + ".tsv")
        if os.path.exists(os.path.join(args.data, args.split) + ".itl"):
            copyfile(os.path.join(args.data, args.split) + ".itl", dest + ".itl")
        if os.path.exists(os.path.join(args.data, args.split) + ".lis"):
            copyfile(os.path.join(args.data, args.split) + ".lis", dest + ".lis")

        if os.path.exists(dest + ".npy"):
            os.remove(dest + ".npy")
        npaa = NpyAppendArray(dest + ".npy")
        return npaa

    save_path = os.path.join(args.save_dir, args.split)
    npaa = create_files(save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    checkpoint_path = args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)

    generator, num = get_iterator(args, model, cfg)
    iterator = generator()

    with open(save_path + ".lengths", "w") as l_f:
        for feats in tqdm.tqdm(iterator, total=num):
            if len(feats.shape) == 2:
                feats = np.repeat(np.expand_dims(feats, axis=1), repeats=2, axis=1)
            print(len(feats), file=l_f)

            if len(feats) > 0:
                npaa.append(np.ascontiguousarray(feats))
    del model


if __name__ == "__main__":
    # import ptvsd
    # ptvsd.enable_attach(('0.0.0.0', 7310))
    # print("Attach debugger now")
    # ptvsd.wait_for_attach()
    main()
