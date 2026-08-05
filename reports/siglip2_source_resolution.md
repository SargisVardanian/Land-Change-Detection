# SigLIP-2 source resolution

Pinned repository: google/siglip2-base-patch16-256.
Revision: 3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab.
License metadata: Apache-2.0.

The checkpoint passed safetensors.safe_open and local CPU inference. The native image output is [1,256,768] and text output is [1,64,768].

The pinned config declares model_type=siglip and the patch embedding is Conv2d [768,3,16,16]. Transformers 5.12.1 Siglip2Model expects a different embedding shape, so AutoModel resolves the checkpoint to SiglipModel. This is an audited runtime resolution; no ignore_mismatched_sizes fallback is used.

GeoRSCLIP is pinned to Zilun/GeoRSCLIP revision 4920188e6eba4e711ef9848cfd7cb77e874ee33f. RS5M_ViT-B-32.pt passed torch.load(weights_only=True) and strict OpenCLIP ViT-B/32 compatibility with zero missing/unexpected keys. Its license metadata is cc and exact weight/data redistribution terms are not established, so it remains a frozen research-only baseline.
