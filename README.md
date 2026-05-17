# Hyperspectral image classification via Manhattan self-attention transformer and adaptive global-local channel attention 2026 JKSUCIS

PyTorch implementation of Hyperspectral image classification via Manhattan self-attention transformer and adaptive global-local channel attention.

# Basic Usage

```
import torch
from MTACANet import MTACANet
model = MTACANet(in_chans=64, num_classes=16, embed_dim=64, num_heads=4, ffn_dim=96)
model.eval()
print(model)
input = torch.randn(100, 64, 13, 13)
y = model(input)
print(y.size())
```

# Paper

[Hyperspectral image classification via Manhattan self-attention transformer and adaptive global-local channel attention](https://link.springer.com/article/10.1007/s44443-026-00484-1)

If you find this code to be useful for your research, please consider citing.

```
@article{meng2026hyperspectral,
  title={Hyperspectral image classification via Manhattan self-attention transformer and adaptive global-local channel attention},
  author={Meng, Zhe and Yue, Pan and Zhao, Feng},
  journal={Journal of King Saud University Computer and Information Sciences},
  volume={38},
  pages={97},
  year={2026},
  publisher={Springer}
}
```


