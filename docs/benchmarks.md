# Benchmarks (Stage I, CPU prototype)

All numbers below were measured on a PC **CPU**. Nothing here is measured on the ASUS UGen300 yet; UGen300 results will be added in Stage II with the same clips and settings (`--backend hailo`).

## Test setup

| Item | Value |
|---|---|
| CPU | 11th Gen Intel Core i5-11320H @ 3.20 GHz (4 cores / 8 threads) |
| RAM | 8 GB |
| Accelerator | none used (onnxruntime `CPUExecutionProvider`) |
| Software | Python 3.12.3, rtmlib 0.0.16, onnxruntime 1.31.0 |
| Models | YOLOX-m (person detection) + RTMPose-m (17-point pose), `--mode balanced` |
| Settings | defaults from `configs/default.yaml`: pose every 2nd frame (`detect_every: 2`), inference long side 960 px, output 1920×1080 |
| Input | 1080p, 60 fps clips cut from one fixed-camera stage performance; 30 s each (clips A–D), demo2 is 28 s |

## Results per clip

Latency is the per-frame time from reading a frame to writing the output frame. Frames that run pose detection are slow and the others are cheap, so the median over all frames is lower than the detection cost.

| Run | Frames | FPS | Latency median / p95 (ms) | Detection median / p95 (ms) | Zoom median (range) |
|---|---:|---:|---:|---:|---|
| clip A: 5 dancers, warm / white spotlights | 1800 | 5.77 | 144 / 430 | 292 / 469 | 2.14 (1.22–2.30) |
| clip B: 8 dancers, then solo + 2 kneeling | 1800 | 5.68 | 227 / 428 | 298 / 457 | 1.48 (1.12–2.12) |
| clip C: 6 dancers, blue light | 1800 | 4.62 | 168 / 680 | 311 / 690 | 2.15 (1.22–2.30) |
| clip D: 7 dancers, purple backlight | 1800 | 5.30 | 156 / 499 | 307 / 585 | 1.97 (1.22–2.23) |
| demo2: 5 dancers, same group, earlier segment (28 s) | 1703 | 5.78 | 260 / 435 | 286 / 472 | 2.15 (1.22–2.30) |
| live A (debug overlay on, 23 s) | 1380 | 5.48 | 160 / 457 | 293 / 478 | 2.21 (1.22–2.30) |

`demo1` (a 360p source upscaled to 1080p) ran at 3.99 FPS, but it ran while ffmpeg was cutting other clips on the same machine, so it is not comparable and is left out of the table.

## Where the time goes

Median per-stage time over all runs above:

| Stage | Median (ms) |
|---|---:|
| Person detection + pose (every 2nd frame) | 297.9 |
| Lighting restoration | 3.5 |
| Crop and upscale | 3.5 |
| Encode and write | 2.5 |
| Compose output view | 2.1 |
| Framing (target, virtual camera) | 0.2 |

Detection is about 95% of the per-frame cost, which is why it is the stage planned to move to the UGen300 NPU.

## Stage II targets

| Metric | CPU prototype (now) | Stage II target (UGen300) |
|---|---|---|
| Throughput at 1080p | ~4.6–5.8 FPS | ≥ 30 FPS |
| End-to-end latency | ~144–260 ms median (all frames) | ≤ 100 ms |

## Reproduce

```bash
# one run, per-stage timings written to a CSV
python src/main.py --input samples/clip_A_1m54.mp4 --view split --no-show \
  --set timings_csv=out/timings_clip_A.csv --output out/clip_A_split.mp4

# the benchmark script (framing only vs. framing + lighting, summary JSON / Markdown)
python scripts/benchmark.py --input samples/clip_A_1m54.mp4 --max-frames 600
```

The table above was produced from `src/main.py` runs, not from `scripts/benchmark.py`.

## What was and was not verified

- **Framing quality** was checked visually on sampled frames: 20 evenly spaced frames each for clips A and C, plus several full-resolution spot checks. No cut-off heads were found in those samples. This is not a frame-by-frame audit.
- **Lighting restoration** was not quantified on stage-wide footage. In clip comparisons about 1% of pixels changed noticeably, because the performers' faces are small and not over-exposed in these clips. The effect needs over-exposed, closer footage to be shown.
- **Known limit:** when a lead performer stands among side performers (clip B, second half), the crop stays wide. Lead-performer selection is planned for Stage II.
