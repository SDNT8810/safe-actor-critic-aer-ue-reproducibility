# Windows notes

All Markdown, LaTeX, JSON, HTML, and CSV outputs are written explicitly as UTF-8, so Windows `cp1252` defaults do not affect arrows or ± symbols.

Install the complete environment with:

```bat
python -m pip install -r requirements.txt
```

`imageio-ffmpeg` provides a portable FFmpeg executable for MP4 generation. GIF generation uses Pillow and can be slower than MP4 generation.

A completed numerical run can be postprocessed without retraining:

```bat
python run.py postprocess --out outputs/reproduction --animate
```
