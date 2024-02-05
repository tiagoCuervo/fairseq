# import json
import torch
import os
import numpy as np
import fairseq
import argparse
import soundfile as sf
from shutil import copyfile
from npy_append_array import NpyAppendArray
import tqdm
from omegaconf import OmegaConf
from fairseq.models.hubert.hubert import HubertConfig, HubertModel


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data", help="Path with .csv files pointing to the audio data")
    parser.add_argument("--split", help="Which split", required=True)
    parser.add_argument("--save-dir", help="Output path to store the features", required=True)
    parser.add_argument("--checkpoint", help="Path to the WavLM checkpoint", required=True)
    return parser


def get_iterator(args, mdl, cfg):
    with open(os.path.join(args.data, args.split) + ".csv", "r") as fp:
        lines = fp.read().split("\n")
        lines.pop(0)
        root = os.path.join(args.data, "confusionWavs")
        # first column is for index, second is for token ID which allows to rebuild the waveform filename
        files = [os.path.join(root, f'T_{line.split(",")[1]}.wav') for line in lines if len(line) > 0]
        num = len(files)

        def iterate():
            for fname in files:
                # feats = reader.get_feats(fname)
                wav, sr = sf.read(fname)
                wav = torch.from_numpy(wav).float().cuda()
                if wav.dim() > 1:
                    wav = torch.mean(wav, dim=1)    
                wav = wav.view(1, -1)
                with torch.no_grad():
                    if cfg.task.normalize:
                        wav = torch.nn.functional.layer_norm(wav , wav.shape)
                    audio_rep = mdl(source=wav, mask=False, features_only=True, output_layer=cfg.model.encoder_layers)
                    audio_rep = audio_rep.squeeze(0).cpu().numpy()
                yield audio_rep
    return iterate, num


def main():
    parser = get_parser()
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    def create_files(dest):
        copyfile(os.path.join(args.data, args.split) + ".csv", dest + ".csv")
        if os.path.exists(os.path.join(args.data, args.split) + ".itl"):
            copyfile(os.path.join(args.data, args.split) + ".itl", dest + ".itl")
        if os.path.exists(os.path.join(args.data, args.split) + ".lis"):
            copyfile(os.path.join(args.data, args.split) + ".lis", dest + ".lis")

        if os.path.exists(dest + ".npy"):
            os.remove(dest + ".npy")
        npaa = NpyAppendArray(dest + ".npy")
        return npaa

    save_path = os.path.join(args.save_dir, 'hubert.'+args.split)
    npaa = create_files(save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    checkpoint_path = args.checkpoint
    # model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    #     [checkpoint_path]
    # )
    # model = model[0]

    checkpoint = torch.load(checkpoint_path, map_location=device)    
    cfg = OmegaConf.create(checkpoint["cfg"])
    if cfg.model._name == "hubert_ctc":
        cfg = cfg.model.w2v_args
        # Create a new state_dict with modified keys
        state_dict = {}
        prefix = "w2v_encoder.w2v_model."

        for key, value in checkpoint['model'].items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                state_dict[new_key] = value
        cfg.model["layer_type"] = "transformer"
        cfg.model["required_seq_len_multiple"] = 1
    else:
        state_dict = checkpoint['model']
    model = HubertModel(HubertConfig.from_namespace(cfg.model), cfg.task, [None])
    model.load_state_dict(state_dict, strict=False)

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
