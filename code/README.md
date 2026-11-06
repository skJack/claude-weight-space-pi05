# Code

The scripts for the main steps, with machine-specific paths replaced by placeholders. Comments are mostly in Chinese.

| File | What it does |
|---|---|
| `pair_frames.py` | Restores simulator states from the training demonstrations and pairs each fresh render with the decoded training-video frame |
| `estimate_directions.py` | Estimates the three directions from those pairs and writes them in the format of `edit/edit.npz`. Rewritten for this release with a pluggable model loader. We ran it on the original pairs: cosine with the published directions is 1.000000 for all three |
| `mk_col.py` | Builds the single-column edits (column 725 and the random-column controls) |
| `official_chain.sh`, `official_client.py`, `serve_rc.py` | The check against the official evaluation stack |
| `check_pairs.py` | Compares the first frame and initial simulator state of every episode pair |
| `render_clip.py`, `render_all_clips.sh`, `export_image_pairs.py` | The clips and image pairs on the page, re-rendered from recorded simulator states and checked against the recordings |

Not included: the closed-loop rollout harness and the policy-loading code. Except for `estimate_directions.py`, these scripts won't run as they stand. To reproduce the headline numbers you need your own rollout loop that follows `examples/robocasa/main.py` in [robocasa-benchmark/openpi](https://github.com/robocasa-benchmark/openpi), plus `edit/apply_edit.py`.
