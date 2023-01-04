# Stable Diffusion Modification Files

This directory contains files needed to augment or modify either the main Stable Diffusion repo or variations:

- txt2img.py
  - This file contains a modified script for the txt2img file in SD which allows for configurations to be submitted through a JSON config file rather than as a CLI argument
- txt2img-gpu-optimized.py
  - This file contains a modified script for the txt2img file in SD Optimized which allows for configurations to be submitted through a JSON config file rather than as a CLI argument
- ddpm-gpu.py
  - Modification of this file to allow for relative imports according to the new directory structure
- openaimodelSplit.py
  - Similar to ddpm, this just modifies the script to operate in the new directory structure
- v1-inference-optimized.yml
  - Tweaked model config script for also handling new the directory structure.

