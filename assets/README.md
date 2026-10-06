# README figures

Place the three PNG files below in this directory. The root `README.md` already
references these exact filenames; adding the files is sufficient to display them
on GitHub. Filenames are case-sensitive on GitHub.

The figure numbers and page numbers refer to *Identifiable Face to Face for
Privacy Protection*, DOI: `10.1109/TIFS.2026.3737713` (the 13-page paper supplied
with this project).

| Filename | Paper source | README location | Content to retain |
|:--|:--|:--|:--|
| `overview.png` | Fig. 1, page 1 | Overview | Local client, original and self-transformed faces, PPIDs, and online registration/authentication |
| `architecture.png` | Fig. 2, page 3 | Overview | Complete IF2F pipeline, including the inversion encoder, RFH, MCLT, generator, and user/cloud sides |
| `qualitative_results.png` | Fig. 5, page 7 | Self-transformed face examples | All five identity columns and six image rows, with Person and Original/Self-Transformed labels |

## Preparation

- Prefer exports from the original figure sources for sharp text and face images.
  If exporting from the paper, crop to the figure itself and keep its internal
  labels. The root README supplies an English caption for each figure, so the
  paper's caption and surrounding page text need not be included in the image.
- Use PNG, retain the original aspect ratio, and avoid unnecessary blank margins.
- For the wide architecture figure, aim for at least 1600 pixels
  in width. For the overview and identity-consistency figures, about 1000-1200
  pixels in width is a useful starting point. Check that labels remain legible in
  GitHub's rendered README.
- Keep the panel and row ordering listed above so that the README captions remain
  accurate. If changing the composition, update the corresponding caption too.

The repository layout after adding the images should be:

```text
assets/
├── README.md
├── overview.png
├── architecture.png
└── qualitative_results.png
```

These are documentation assets only and are not loaded by training or inference.
