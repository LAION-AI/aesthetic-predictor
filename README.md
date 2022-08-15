# LAION-Aesthetics_Predictor V1

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LAION-AI/aesthetic-predictor/blob/main/asthetics_predictor.ipynb)

A linear estimator on top of clip to predict the aesthetic quality of pictures

Normal results:

<img src="cat_normal.png" width="512" />

Aesthetic results:

<img src="cat_aesthetic.png" width="512" />

visualization over laion2B http://3080.rom1504.fr/aesthetic/aesthetic_viz.html

integrated into https://rom1504.github.io/clip-retrieval/

```python
import os
import torch
import torch.nn as nn
from os.path import expanduser  # pylint: disable=import-outside-toplevel
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel
def get_aesthetic_model(clip_model="vit_l_14"):
    """load the aethetic model"""
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m
```

That model takes clip embeddings as input 

https://github.com/rom1504/embedding-reader/blob/main/examples/aesthetic_inference.py
https://github.com/rom1504/embedding-reader/blob/main/examples/aesthetic_inference_emb.py
