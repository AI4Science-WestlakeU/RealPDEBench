
<p align="center">
  <img src="imgs/logo.png" alt="RealPDEBench logo" width="700" />
</p>

# RealPDEBench: A Benchmark for Complex Physical Systems with Paired Real-World and Simulated Data

[![HF Dataset](https://img.shields.io/badge/HF%20Dataset-RealPDEBench-FFD21E?logo=huggingface)](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDEBench)
[![arXiv](https://img.shields.io/badge/arXiv-2601.01829-b31b1b?logo=arxiv)](https://arxiv.org/abs/2601.01829)
[![Website & Docs](https://img.shields.io/badge/Website%20%26%20Docs-realpdebench.github.io-1f6feb?logo=google-chrome)](https://realpdebench.github.io/)
[![Codebase](https://img.shields.io/badge/Codebase-GitHub-181717?logo=github)](https://github.com/AI4Science-WestlakeU/RealPDEBench)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-9cf?logo=creativecommons&logoColor=white)](https://creativecommons.org/licenses/by-nc/4.0/)

[Peiyan Hu](https://peiyannn.github.io/)<sup>∗†1,3</sup>, [Haodong Feng](https://scholar.google.com/citations?user=0GOKl_gAAAAJ&hl=en)<sup>*1</sup>, [Hongyuan Liu](https://orcid.org/0009-0007-0168-0510)<sup>*1</sup>, Tongtong Yan<sup>2</sup>, Wenhao Deng<sup>1</sup>, Tianrun Gao<sup>†1,4</sup>, Rong Zheng<sup>†1,5</sup>, Haoren Zheng<sup>†1,2</sup>, Chenglei Yu<sup>1</sup>, Chuanrui Wang<sup>1</sup>, Kaiwen Li<sup>†1,2</sup>, Zhi-Ming Ma<sup>3</sup>, Dezhi Zhou<sup>2</sup>, Xingcai Lu<sup>6</sup>, Dixia Fan<sup>1</sup>, [Tailin Wu](https://tailin.org/)<sup>†1</sup>.<br />

<sup>1</sup>School of Engineering, Westlake University; 
<sup>2</sup>Global College, Shanghai Jiao Tong University;
<sup>3</sup>Academy of Mathematics and Systems Science, Chinese Academy of Sciences;
<sup>4</sup>Department of Geotechnical Engineering, Tongji University; 
<sup>5</sup>School of Physics, Peking University;
<sup>6</sup>Key Laboratory for Power Machinery and Engineering of M. O. E., Shanghai Jiao Tong University<br />

</sup>*</sup>Equal contribution, </sup>†</sup>Work done as an intern at Westlake University, </sup>†</sup>Corresponding authors

------


## 💧🔥 Overview

RealPDEBench is the first scientific ML benchmark with **paired real-world measurements and matched numerical simulations**
for complex physical systems, designed for **spatiotemporal forecasting** and **sim-to-real transfer**.

**At a glance 👀**
- **[5 Datasets](https://realpdebench.github.io/datasets/)**: `cylinder`, `fsi`, `controlled_cylinder`, `foil`, `combustion`
- **[700+ Trajectories](https://realpdebench.github.io/datasets/#dataset-inventory)**
- **[10 Baseline models](https://realpdebench.github.io/models/)**: U-Net, FNO, CNO, WDNO, DeepONet, MWT, GK-Transformer, Transolver, DPOT, DMD
- **[9 Evaluation metrics](https://realpdebench.github.io/metrics/)**: RMSE, MAE, Rel L₂, R², Update Ratio, fRMSE, FE, KE, MVPE

------

## 🎬 Installation (pip)

This repo is packaged with `pyproject.toml` and can be installed via pip (requires Python ≥ 3.10):

```bash
git clone https://github.com/AI4Science-WestlakeU/RealPDEBench.git
cd RealPDEBench
pip install -e .
```

------


## ⏬ Dataset download


<a href="url"><img src="https://github.com/AI4Science-WestlakeU/RealPDEBench/blob/main/imgs/figure1.png" align="center" width="1000" ></a>

### Hugging Face dataset: 

The repo id `AI4Science-WestlakeU/RealPDEBench`.

We provide a small pattern-based downloader:

```bash
# safe default: download metadata JSONs only
realpdebench download --dataset-root /path/to/data --scenario fsi --what metadata

# to download Arrow shards (LARGE), explicitly set --what=hf_dataset or --what=all
# splits are stored in index JSONs under hf_dataset/ (no split directories)
realpdebench download --dataset-root /path/to/data --scenario fsi --what hf_dataset --dataset-type real
```

Tips:
- Set `--endpoint https://hf-mirror.com` (or env `HF_ENDPOINT`) to get acesss.
- If you hit rate limits (HTTP 429) or need auth, login and set env `HF_TOKEN=...`.
- We recommend setting env `HF_HUB_DISABLE_XET=1`.

### HDF5-format dataset

Coming soon!

------

## 📝 Checkpoint & log file download

Coming soon!

------

## 📥 Training

```bash
# Simulated training (train on numerical data)
python -m realpdebench.train --config configs/cylinder/fno.yaml --train_data_type numerical

# Real-world training (train on real data from scratch)
python -m realpdebench.train --config configs/cylinder/fno.yaml --train_data_type real

# Real-world finetuning (finetune on real data)
python -m realpdebench.train --config configs/cylinder/fno.yaml --train_data_type real --is_finetune
```

### Using HF Arrow-backed datasets

HF Arrow datasets are stored under `{dataset_root}/{scenario}/hf_dataset/{real,numerical}/` with split index files
`{split}_index_{type}.json`. To use them, enable:
- `--use_hf_dataset`: load Arrow trajectories + index files (lazy slicing, dynamic `N_autoregressive`)
- `--hf_auto_download`: download missing artifacts from HF automatically (use `--hf_endpoint https://hf-mirror.com` for easy accessing)

Example:

```bash
python -m realpdebench.train --config configs/fsi/fno.yaml --use_hf_dataset --hf_auto_download --hf_endpoint https://hf-mirror.com
```

------

## 📤 Inference

```bash
python -m realpdebench.eval --config configs/fsi/fno.yaml --checkpoint_path /path/to/checkpoint.pth
```

------

## Numerical scripts

Coming soon!

------

## 👩‍💻 Contribute

We welcome contributions from the community! Please feel free to 

- [Add your models](https://realpdebench.github.io/models/add-your-model/)
- Contact us to submit to the [leaderboard](https://realpdebench.github.io/leaderboard/)
- Contribute code improvements
- Improve documentation

------

## 🫡 Citation

If you find our work and/or our code useful, please cite us via:

```bibtex
@misc{hu2026realpdebenchbenchmarkcomplexphysical,
      title={RealPDEBench: A Benchmark for Complex Physical Systems with Real-World Data}, 
      author={Peiyan Hu and Haodong Feng and Hongyuan Liu and Tongtong Yan and Wenhao Deng and Tianrun Gao and Rong Zheng and Haoren Zheng and Chenglei Yu and Chuanrui Wang and Kaiwen Li and Zhi-Ming Ma and Dezhi Zhou and Xingcai Lu and Dixia Fan and Tailin Wu},
      year={2026},
      eprint={2601.01829},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.01829}, 
}
```

------

## 📚 Related Resources

- AI for Scientific Simulation and Discovery Lab: https://github.com/AI4Science-WestlakeU
- REALM: https://github.com/deepflame-ai/REALM/tree/main
- PDEBench: https://github.com/pdebench/PDEBench
