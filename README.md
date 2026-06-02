# FiNE: Fine-grained Neuron-level Model Editing for Reliable and Safe LLMs

This repository is the official implementation of the paper ["FiNE: Fine-grained Neuron-level Model Editing for Reliable and Safe LLMs"](https://opanhw.github.io/fine-unified-editing/)


## Repository layout

| Directory | Role |
|-----------|------|
| `mm-neurons/` | Multimodal model analysis |
| `knowledge-editing/` | Knowledge editing |
| `safety-editing/` | Safety editing |

## Requirements

```bash
git clone https://github.com/opanhw/fine-unified-editing-code.git
cd fine-unified-editing-code
conda create -n unified-editing python=3.11
conda activate unified-editing
pip install -r requirements.txt
```

The main `requirements.txt` is intended for **knowledge-editing** and **safety-editing**. If you only use **mm-neurons**, use a separate environment and the optional pins at the bottom of `requirements.txt` (they target an older `transformers` release).

---

## Multi-modal neurons (`mm-neurons/`)

All commands below are run from **`mm-neurons/`** unless noted.

### Model preparation

We support [LLaVA](https://github.com/haotian-liu/LLaVA), [InstructBLIP](https://github.com/salesforce/lavis), and [mPLUG-Owl2](https://github.com/X-PLUG/mPLUG-Owl). Download weights from Hugging Face, for example:

- [LLaVA](https://huggingface.co/liuhaotian/llava-llama-2-13b-chat-lightning-preview)
- [InstructBLIP](https://huggingface.co/Salesforce/instructblip-vicuna-7b)
- [mPLUG-Owl2](https://huggingface.co/MAGAer13/mplug-owl2-llama2-7b)

**Please note:** we recommend LLaVA [v1.0.2](https://github.com/haotian-liu/LLaVA/archive/refs/tags/v1.0.2.tar.gz); newer releases may diverge from the class interfaces assumed here.

Patch upstream sources so activations can be read and edited: copy `open_source_model/LLaVA/llava_llama.py` into your LLaVA tree (`llava/model/language_model/llava_llama.py`), and `open_source_model/mPLUG-Owl2/modeling_mplug_owl2.py` into mPLUG-Owl2 (`mplug_owl2/model/modeling_mplug_owl2.py`).

Run preparation once:

```bash
cd mm-neurons
python src/preparation.py --model_type LLaVA --model_path YOUR_LLAVA_MODEL_PATH
```

`src/preparation.py` and `src/trainer.py` add `open_source_model/...` to `sys.path` relative to the `mm-neurons` tree (not the current working directory).

### Dataset

- **SBU Captions** — description [here](https://www.cs.rice.edu/~vo9/sbucaptions/); download the [json archive](https://www.cs.rice.edu/~vo9/sbucaptions/sbu-captions-all.tar.gz).

### Finding multi-modal neurons

```bash
cd mm-neurons
python src/find_mm_neurons.py --model_type LLaVA --model_path YOUR_LLAVA_MODEL_PATH --save_to ./results --task_type sbu --data_path ./datasets/sbu --add_prompt --cal_activations
```

Arguments:

- `model_type`: `LLaVA`, `InstructBLIP`, or `mPLUG-Owl2` (case-insensitive)
- `model_path`: checkpoint path
- `save_to`: output directory
- `task_type`: dataset tag; `sbu` is supported
- `data_path`: dataset root
- `query`: prompt (we use `Describe the image in few words.` in experiments)
- `max_num`, `start`, `end`: subsampling (defaults 0–1000)
- `add_prompt`: prepend `"An image of"`
- `cal_activations`: dump activations
- `shuffle`: shuffle image-token order

For LLaVA heatmaps / binary masks, replace `transformers/.../blip/image_processing_blip.py` in your environment with `src/image_processing_blip.py` so tensors are visible before normalization.

---

## Knowledge editing (`knowledge-editing/`)

Run scripts from **`knowledge-editing/`** so `./hparams`, `./datasets`, and `./results` resolve correctly.

### Benchmark

- **KnowEdit** — see the dataset card on [Hugging Face](https://huggingface.co/datasets/zjunlp/KnowEdit).

### Editing

Configure hyperparameters before running. Method-specific YAMLs live under `./hparams/FINE/`, including for example:

- `model_name`: local path or Hugging Face model id
- `epochs`, `lr`, `alpha`, `beta`, `gamma`
- `neuron_num`, `early_stop_prob`, `last_layer`, `fp16`, and related fields

Some legacy YAMLs under other `hparams/` subfolders may still contain example absolute paths; replace them with your machine’s paths or Hugging Face ids.

Example drivers:

```bash
cd knowledge-editing
bash run_gptj_fine.sh
bash run_llama2_fine.sh
bash run_llama3_fine.sh
```

---

## Safety editing (`safety-editing/`)

- **`easyeditor/`** — Python package (editing methods, trainers, utilities).
- **`hparams/`** — per-method YAML; fill `model_name`, and where needed `stats_dir` / `P_loc` (placeholders are empty with `USER` comments).
- **`scripts/`** — `run_safety_editing.py`, `run_safety_editing_eval.py`, `run_safety_editing_pre.py`, `opencompass_eval.py`.
- **`shell/`** — Bash examples; set the path variables at the top before execution.

Install from the repo root as above. When calling the Python scripts, you **must** pass non-empty `--safety_classifier_dir`, `--data_dir`, and `--metrics_save_dir` (empty defaults exit with an error). Each script inserts the `safety-editing` root into `sys.path` so `import easyeditor` works without extra `PYTHONPATH`.

For OpenCompass, edit `safety-editing/scripts/opencompass_eval.py` to set the model `path` and `work_dir`, then use `safety-editing/shell/opencompass_eval.sh` as a template.

---

## Acknowledgements

Parts of this codebase build on public software and research tooling, including [EasyEdit](https://github.com/zjunlp/EasyEdit) and [Skill-Neuron](https://github.com/THU-KEG/Skill-Neuron)

## Citation

If you find this code useful, please kindly cite our work as:

```bibtex
@article{yang2026fine,
  title={FiNE: Fine-grained Neuron-level Model Editing for Reliable and Safe LLMs},
  author={Yang, Xun and Pan, Haowen and Wang, Xiaozhi and Cao, Yixin and Li, Juanzi and Wang, Meng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2026},
  publisher={IEEE}
}
```
