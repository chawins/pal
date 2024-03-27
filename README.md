# PAL: Proxy-Guided Black-Box Attack on Large Language Models

Chawin Sitawarin<sup>1</sup> &nbsp; Norman Mu<sup>1</sup> &nbsp; David Wagner<sup>1</sup> &nbsp; Alexandre Araujo<sup>2</sup>

<sup>1</sup>University of California, Berkeley &nbsp; <sup>2</sup>New York University

> UPDATE (March 17, 2024): This [change](https://twitter.com/brianryhuang/status/1763438814515843119) made by OpenAI may affect the success of this attack.

## Abstract

Large Language Models (LLMs) have surged in popularity in recent months, but they have demonstrated concerning capabilities to generate harmful content when manipulated. While techniques like safety fine-tuning aim to minimize harmful use, recent works have shown that LLMs remain vulnerable to attacks that elicit toxic responses. In this work, we introduce the Proxy-Guided Attack on LLMs (PAL), the first optimization-based attack on LLMs in a black-box query-only setting. In particular, it relies on a surrogate model to guide the optimization and a sophisticated loss designed for real-world LLM APIs. Our attack achieves 84% attack success rate (ASR) on GPT-3.5-Turbo and 48% on Llama-2-7B, compared to 4% for the current state of the art. We also propose GCG++, an improvement to the GCG attack that reaches 94% ASR on white-box Llama-2-7B, and the Random-Search Attack on LLMs (RAL), a strong but simple baseline for query-based attacks. We believe the techniques proposed in this work will enable more comprehensive safety testing of LLMs and, in the long term, the development of better security guardrails.

## Install Dependencies

Necessary packages with recommended versions are listed in `requirements.txt`. Run `pip install -r requirements.txt` to install these packages.

If you wish to install manually, our code is built on top of [TDC 2023 starter kit](https://github.com/centerforaisafety/tdc2023-starter-kit/tree/main/red_teaming).
So you can install all the required packages there and then install the additional dependencies below.

```bash
pip install python-dotenv anthropic tenacity google-generativeai num2words bitsandbytes tiktoken sentencepiece torch_optimizer
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes
# Our code is not tested with transformers >= 4.36
pip install transformers==4.35.2 fschat==0.2.34
```

## Example

```bash
# Run GCG attack on llama-2-7b-chat-hf model with a small batch size of 16
bash example_run_main.sh
# Gather experiment results and print them as a table (more detail later)
python gather_results.py
```

- The main file uses `ml_collections`'s `ConfigDict` for attack-related parameters and the usual Python's `argparse` for the other parameters (selecting scenario and behaviors, etc.).
- Each attack comes with its own config file in `./configs/ATTACK_NAME.py`.
- `--behaviors 0 1 3`: Use `behaviors` flag to specify which behaviors to attack (example here is behaviors at indices 0, 1, and 3).

### Where to find the attack results

- Log path is given by `./results/<MODEL>/<ATTACK>/<EXP>/<SCENARIO>_<BEHAVIOR>.jsonl`. Example: `./results/Llama-2-7b-chat-hf/ral/len20_100step_seed20_static_bs512_uniform_t1.0_c8-1/Toxicity_0.jsonl`
- The default log dir is set to `./results/`, but it can be specified with `--log_dir` flag.
- `<ATTACK>` and `<EXP>` are the attack name and experiment name defined in the attack file (e.g., `./src/attacks/gcg.py`). See `_get_name_tokens()`.

### Reproducibility

**When the random seed is set the following step is unnecessary.**
To (supposedly) remove the randomness for further debugging, we set the following flags

In the bash script or before running `main.py`:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

In `main.py`:

```python
torch.use_deterministic_algorithms(True)
```

Even when these two are enabled, we observe a slight difference between gradients computed with and without KV-cache (may also be due to half precision).
For example, in the GCG attack, the order of the top-k can shift slightly when k is large, but overall most of the tokens are the same.
This can result in a different adversarial suffix.

## Code Structure

### Main Files

- `main.py`: Main file for running attacks.
- `gather_results.py`: Gather results from log files and print them as a table.

### `Src`

Most of the attack and model code is in `src/`.

- `attacks` contains all the attack algorithm. To add a new attack, create a new file in this directory and import and add your attack to `_ATTACKS_DICT` in `attacks/__init__.py`. We highly recommend extending `BaseAttack` class in `attacks/base.py` for your attack. See `attacks/gcg.py` or `attacks/ral.py` for examples.
  - `attacks/gcg.py`: contains our GCG++ which is built from a minimal version of the original GCG attack ([code](https://github.com/llm-attacks/llm-attacks), [paper](https://arxiv.org/abs/2307.15043)).
  - `attacks/ral.py`: Our RAL attack.
  - `attacks/pal.py`: Our PAL attack.
- `models` contains various model interfaces.
- `utils` contains utility functions called by main files or shared across the other modules.

## Attacks

### PAL Attack

To fine-tune the proxy model, `config.finetune=True`. Below are the available fine-tuning options.

- Fine-tune with pure `bfloat16`: `config.pure_bf16=True`. This is recommended and uses much less memory than `float16`.
- Fine-tune with mixed precision (`float16`): `config.use_fp16=True`. Both `use_fp16` and `pure_bf16` cannot be `True` at the same time.
- Fine-tune with PEFT: `config.use_peft=True`. This is not compatible with `use_fp16` or `pure_bf16` (yet).
- Fine-tune with PEFT and quantization (`int8`): `config.use_peft=True` and `config.quantize=True`.

Notes

- Use a larger learning rate when fine-tuning with PEFT (e.g., `1e-3`).
- For 7B models and `pure_bf16` on one A100, `config.mini_batch_size <= 128` and `config.proxy_tune_bs < 64`. `proxy_tune_bs` of 64 will fail on some longer prompts. Use `proxy_tune_bs` of 32 to be safe.
- Cannot train a 7B model with `use_fp16` on one A100 even with a batch size of 1.

### OpenAI API

- Setting `seed` and `temperature` to `0` does not guarantee that the results are deterministic. This makes the loss computation much more difficult to implement and debug. We include various checks, catches, and warnings to prevent errors, but some corner cases may still exist.
