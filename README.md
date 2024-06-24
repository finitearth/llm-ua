# LLM Uncertainty Attribution
This repo provides tools to compute the feature attribution maps for Laplace-approximated LLMs. Given a Laplace-approximated model,  the attribution can be conducted on any intermediate layer or input tokens. Additionally, ithe repo contains an easy-to-use Flask app for visualizing the obtained attribution maps.

![grafik](https://github.com/finitearth/llm-ua/assets/19229952/39eadabd-2f3e-4c1b-89ad-8ed2355d310d)

## Installation
1. Install `torch` following the documentations of `PyTorch`.
2. Install `flash-attn` via `FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install flash-attn --no-build-isolation`.
2. Install the package in editable mode via `pip install -e .` (do no forget the "."). For Developers,
install the package along with extra dependencies via `pip install -e ".[dev]"`.

## Optional
Install pre-commit hooks via `pre-commit install`.

## Fine-tuning and Laplace Approximation
run `python tools/fine_tune.py configs/ft_llama-2_arc-c.yaml -w workdirs/debug` to fine-tune the model and run the laplace approximation or check python file for details.

## Attribution
`python tools/attribution_cli.py configs/attr_llama-2_arc-c.yaml -w workdirs/debug/` or check the python file for details.

## Flask app for showing attribution maps
The app requires `Flask`, which is already listed in `"dev"` feature in `setup.cfg`.
`python tools/app.py /path/to/vis_attributions/`
and then open `http://127.0.0.1:5000` in your browser.


