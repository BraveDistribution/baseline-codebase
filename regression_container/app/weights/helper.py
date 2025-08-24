import torch


# Load checkpoint
# ckpt1 = torch.load('best-checkpoint-v4.ckpt', map_location='cpu')
# ckpt2 = torch.load('weights.ckpt', map_location='cpu')
# print(ckpt2.keys())
# state_dict = ckpt1['state_dict']

# for name, param in state_dict.items():
#     print(f"{name}: {tuple(param.shape)}")

# print(ckpt2["encoder.swinViT.patch_embed.proj.weight"])
# print(ckpt1["state_dict"]["encoder.swinViT.patch_embed.proj.weight"])
# print(ckpt1["state_dict"].keys() == ckpt2.keys())
# print(torch.equal(ckpt1["state_dict"], ckpt2.keys()))

