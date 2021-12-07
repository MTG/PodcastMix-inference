# Podcastmix: A dataset for separating music and speech in podcasts

This repository contains the instructions for downloading and using the pre-trained UNet and ConvTasNet models in the context of the ICASSP 2022 submission "Podcastmix: A dataset for separating music and speech in podcasts".
If you want to download the complete dataset and train or evaluate your models, please refer to [this](https://github.com/MTG/Podcastmix) repository.

## Install
Create a conda environment:

```
conda env create -f environment.yml
```

Activate the environment:

```
conda activate Podcastmix
```

Download the UNet and ConvTasNet model from the PodcastMix repository:
```
curl https://raw.githubusercontent.com/MTG/Podcastmix/main/UNet_model/unet_model.py -o UNet_model/unet_model.py
curl https://raw.githubusercontent.com/MTG/Podcastmix/main/UNet_model/unet_parts.py -o UNet_model/unet_parts.py
curl https://raw.githubusercontent.com/MTG/Podcastmix/main/ConvTasNet_model/conv_tasnet_norm.py -o ConvTasNet_model/conv_tasnet_norm.py
```

Uncompress the pretrained models and  **overwrite previous file**:

```
zip -F UNet_model/exp/tmp/best_model_splitted.zip --out UNet_model/exp/tmp/best_model.zip
unzip UNet_model/exp/tmp/best_model.zip -d UNet_model/exp/tmp/
zip -F ConvTasNet_model/exp/tmp/best_model_splitted.zip --out ConvTasNet_model/exp/tmp/best_model.zip
unzip ConvTasNet_model/exp/tmp/best_model.zip -d ConvTasNet_model/exp/tmp/
 
```

## Use the model to separate podcasts:

```
[MODEL]
```

could be any of the following:

- ConvTasNet
- UNet (supports 2s and 18s segments)

Without GPU
```
python forward_podcast.py \
    --test_dir=<directory-of-the-podcastmix-real-no-reference-or-your-files> --target_model=[MODEL] \
    --exp_dir=[MODEL]_model/exp/tmp --out_dir=separations \
    --segment=18 --sample_rate=44100 --use_gpu=0
```

With GPU:
```
CUDA_VISIBLE_DEVICES=0 python forward_podcast.py \
    --test_dir=<directory-of-the-podcastmix-real-no-reference-or-your-files> --target_model=[MODEL] \
    --exp_dir=[MODEL]_model/exp/tmp --out_dir=separations \
    --segment=18 --sample_rate=44100 --use_gpu=1
```

Note: the --out_dir folder will be created inside the --exp_dir folder.
