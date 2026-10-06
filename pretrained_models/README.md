# Pretrained model layout

This directory intentionally does not contain large model files. Download the released checkpoint package from [Google Drive](https://drive.google.com/drive/folders/1bWWwn1a26YbUeU_DMuhlnD-KLcd4nQ9p?usp=sharing) or [Baidu NetDisk](https://pan.baidu.com/s/1G05whYX4Zyw2FXPsoP9cvw) (extraction code: `kh14`) and arrange the files as follows:

```text
pretrained_models/
├── best_iter212000.pt
├── e4e_ffhq_encode_256.pt
├── model_ir_se50.pth
├── parsenet.pth
└── stylegan2-ffhq-256.pt
```

`parsenet.pth` is needed only for training. The other four files are required for inference.

Only `best_iter212000.pt` is the released IF2F checkpoint. The auxiliary checkpoints remain attributable to their upstream projects:

| File | Upstream source | Official download |
|:--|:--|:--|
| `e4e_ffhq_encode_256.pt` | [encoder4editing](https://github.com/omertov/encoder4editing) | [FFHQ e4e checkpoint](https://drive.google.com/file/d/1cUv_reLE6k3604or78EranS7XzuVMWeO/view?usp=sharing) |
| `stylegan2-ffhq-256.pt` | [rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch) | [256 px FFHQ checkpoint](https://drive.google.com/file/d/1PQutd-JboOCOZqmd95XWxWrO8gGEvRcO/view) |
| `model_ir_se50.pth` | [TreB1eN/InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch#2-pretrained-models--performance) | Upstream Baidu NetDisk/OneDrive links |
| `parsenet.pth` | [CelebAMask-HQ face parsing](https://github.com/switchablenorms/CelebAMask-HQ/tree/master/face_parsing) | [Original `model.pth`](https://drive.google.com/file/d/1o1m-eT38zNCIFldcRaoWcLvvBtY8S4W3/view) |

The upstream StyleGAN2 and CelebAMask-HQ files may be named `550000.pt` and `model.pth`; rename them to the filenames expected by IF2F. The upstream e4e project publishes `e4e_ffhq_encode.pt`, while IF2F expects the compatible 256-pixel variant shown above. Use the release copy unless compatibility with the 256-pixel, 14-code IF2F pipeline has been verified.

Expected release files:

| File | Size | SHA-256 |
|:--|--:|:--|
| `best_iter212000.pt` | 333.9 MiB | `5f721095a46cc3099b2705acb5ec5e4568b8ec8a5efe77c8d6ade4e69af2813f` |
| `e4e_ffhq_encode_256.pt` | 914.6 MiB | `8588a4e4e8239526329c0905536354cb8c18e1e97956a6f231bac87551587440` |
| `stylegan2-ffhq-256.pt` | 788.7 MiB | `70a891acc3c82280c9a8d132b62e4672e2273d14b692a0df5cb0e416de433c1b` |
| `model_ir_se50.pth` | 167.2 MiB | `a035c768259b98ab1ce0e646312f48b9e1e218197a0f80ac6765e88f8b6ddf28` |
| `parsenet.pth` | 7.5 MiB | `7914f1dccb37ab2f307703f2593efd74cc327d9db974424dff8d7510b46d85b5` |

The IF2F MIT License does not relicense these third-party models. Review and comply with each upstream project's terms; CelebAMask-HQ in particular limits its dataset and released software to non-commercial research or educational use.
