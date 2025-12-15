# Versatile-Diffusion
Versatile-Diffusion rebuild in jittor

## External resources
The training and evaluation pipelines rely on a few external assets that are not distributed in this repository:

- CLIP model checkpoints (e.g., `ViT-B-32`) for text/image feature extraction and CLIPScore computation.
- Pre-computed Inception statistics (such as COCO val FID stats) for FID evaluation.
- LPIPS weights for perceptual similarity evaluation in variation tasks.

The corresponding paths can be configured in the evaluation YAML files (for example, `fid_stats_path`, `clip_model_name`, or `lpips_weight`). Ensure these files are downloaded and the config paths are updated before running training or evaluation.
