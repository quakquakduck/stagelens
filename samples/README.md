# Sample footage

This folder holds the input clips used for the results in `docs/benchmarks.md`. The video files are **not committed** (see `.gitignore`): they are cut from a downloaded stage-performance video, are large, and their licence and performers' consent must be confirmed before public redistribution.

To run StageLens, put any fixed-camera stage video here (or pass any path to `--input`):

```bash
python src/main.py --input samples/your_clip.mp4 --view split --debug
```

Good test footage: fixed camera at the back of the hall, 1 to 6 performers, full bodies in frame, 1080p or better. Very wide groups (more than ~80% of the stage width) fall back to a full frame, and a single lead among side performers is not picked yet (Stage II).

Clips used for the published numbers (not included): clip A (5 dancers, warm / white spotlights), clip B (8 dancers, then solo + 2 kneeling), clip C (6 dancers, blue light), clip D (7 dancers, purple backlight); each is 30 s, 1080p, 60 fps.
