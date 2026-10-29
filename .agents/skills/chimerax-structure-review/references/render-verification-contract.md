# Render Verification Contract

Use this contract after ChimeraX writes a still, numbered frame series, or
encoded movie. A successful ChimeraX command is not sufficient render evidence.

## Required Checks

For stills and sampled movie frames:

- dimensions match the declared output dimensions;
- all four corners match the declared background within a small color tolerance;
- the image contains non-background pixels;
- an optional minimum content extent catches excessive empty framing;
- at least the first, middle, and last movie frames are checked.

For encoded movies:

- dimensions match the frame series;
- frame count matches the declared capture plan;
- duration and frame rate match the encoding contract;
- the movie is encoded only from a frame series that passed the still checks.

Corner validation catches the partial black bands produced by some offscreen
buffers. It does not replace visual inspection of molecule clipping, detached
nucleotide atoms, labels, or surface opacity.

## Helper

Verify a still:

```bash
python .agents/skills/chimerax-structure-review/scripts/chimerax-verify-render.py \
  --image outputs/reference.png \
  --expected-width 1400 \
  --expected-height 1400 \
  --background '#FFFFFF' \
  --minimum-content-extent 0.30
```

Verify representative frames and an encoded movie:

```bash
python .agents/skills/chimerax-structure-review/scripts/chimerax-verify-render.py \
  --image outputs/frames/frame-00001.png \
  --image outputs/frames/frame-00162.png \
  --image outputs/frames/frame-00324.png \
  --movie outputs/structure-story.mp4 \
  --expected-width 1200 \
  --expected-height 1200 \
  --expected-frame-count 324 \
  --expected-duration-seconds 10.8 \
  --background '#FFFFFF'
```

The helper emits JSON and exits nonzero on failure. `ffmpeg` and `ffprobe` are
required because they are the public command-line interfaces used for pixel
sampling and media metadata.
