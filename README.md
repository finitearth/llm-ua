# LLM Uncertainty Attribution

## Installation
1. Install `torch` following the documentations of `PyTorch`.
2. Install `flash-attn` via `FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install flash-attn --no-build-isolation`.
2. Install the package in editable mode via `pip install -e .` (do no forget the "."). For Developers,
install the package along with extra dependencies via `pip install -e ".[dev]"`.

## Optional
Install pre-commit hooks via `pre-commit install`.

## Attribution
`python tools/attribution_cli.py configs/attr_llama-2_arc-c.yaml -w workdirs/debug/` or check the python file for details.

## Flask app for showing attribution maps
The app requires `Flask`, which is already listed in `"dev"` feature in `setup.cfg`.
`python tools/app.py /path/to/vis_attributions/`
and then open `http://127.0.0.1:5000` in your browser.
