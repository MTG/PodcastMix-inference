import argparse
from torch.utils.data.dataset import Dataset
import yaml
import os
import sys
import torch
import soundfile as sf
from asteroid.utils import tensors_to_device
import numpy as np
from tqdm import tqdm
import torchaudio
from utils.my_import import my_import

class PodcastLoader(Dataset):
    dataset_name = "PodcastMix"
    def __init__(self, csv_dir, sample_rate=44100, segment=3):
        self.segment = segment
        self.sample_rate = sample_rate
        self.paths = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if (os.path.isfile(os.path.join(csv_dir, f)) and '.wav' in f)]
        #self.paths = sorted(self.paths, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
        torchaudio.set_audio_backend(backend='soundfile')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        starting_second = 0
        podcast_path = self.paths[index]
        audio_signal, _ = torchaudio.load(
            podcast_path,
            frame_offset=starting_second * self.sample_rate,
            num_frames=self.segment * self.sample_rate,
            normalize=True
        )
        audio_signal = torch.mean(audio_signal, dim=0)
        return audio_signal

    def __getitem_name__(self, index):
        podcast_path = self.paths[index]
        return os.path.splitext(os.path.basename(podcast_path))[0]


parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_dir",
    type=str,
    required=True,
    help="Test directory including the csv files"
)
parser.add_argument(
    "--use_gpu",
    type=int,
    default=0,
    help="Whether to use the GPU for model execution"
)
parser.add_argument(
    "--target_model",
    type=str,
    required=True,
    help="Asteroid model to use"
)
parser.add_argument(
    "--exp_dir",
    default="exp/tmp",
    help="Best serialized model path"
)
parser.add_argument(
    "--out_dir",
    type=str,
    default='ConvTasNet/eval/tmp',
    required=True,
    help="Directory where the eval results" " will be stored",
)
parser.add_argument(
    "--segment",
    type=int,
    default=2,
    required=True,
    help="Number of seconds to separate",
)
parser.add_argument(
    "--sample_rate",
    type=int,
    default=8192,
    required=True,
    help="Sample rate",
)

def main(conf):
    model_path = os.path.join(conf["exp_dir"], "best_model.pth")
    if conf["target_model"] == "UNet":
        sys.path.append('UNet_model')
        AsteroidModelModule = my_import("unet_model.UNet")
    else:
        sys.path.append('ConvTasNet_model')
        AsteroidModelModule = my_import("conv_tasnet_norm.ConvTasNetNorm")
    model = AsteroidModelModule.from_pretrained(model_path, sample_rate=conf["sample_rate"])

    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device
    test_set = PodcastLoader(
        csv_dir=conf["test_dir"],
        sample_rate=conf["sample_rate"],
        segment=conf["segment"]
    )
    eval_save_dir = os.path.join(conf["exp_dir"], conf["out_dir"])
    ex_save_dir = os.path.join(eval_save_dir, "examples_podcast/")
    torch.no_grad().__enter__()
    for idx in tqdm(range(len(test_set))):
        # Forward the network on the mixture.
        mix = test_set[idx]
        mix = tensors_to_device(mix, device=model_device)
        if conf["target_model"] == "UNet":
            est_sources = model(mix.unsqueeze(0)).squeeze(0)
        else:
            est_sources = model(mix)
        mix_np = mix.cpu().data.numpy()
        est_sources_np = est_sources.squeeze(0).cpu().data.numpy()

        # Save some examples in a folder. Wav files and metrics as text.
        local_save_dir = os.path.join(ex_save_dir, "{}/".format(test_set.__getitem_name__(idx)))
        os.makedirs(local_save_dir, exist_ok=True)
        sf.write(
            local_save_dir + "mixture.wav",
            mix_np,
            conf["sample_rate"]
        )
        # Loop over the estimates sources
        for src_idx, est_src in enumerate(est_sources_np):
            est_src *= np.max(np.abs(mix_np)) / np.max(np.abs(est_src))
            sf.write(
                local_save_dir + "s{}_estimate.wav".format(src_idx),
                est_src,
                conf["sample_rate"],
            )

if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))
    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    # arg_dic["segment"] = train_conf["data"]["segment"]
    arg_dic["train_conf"] = train_conf

    main(arg_dic)
