<h1 align="left">Annotator Reliability and Probabilistic Consensus for Semantic Segmentation in Digital Pathology</h1>

<p align="left">
📄 Paper under review | Preprint available at:  
<a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5271896">SSRN Preprint</a>
</p>

---

## 📌 Project Overview

This repository provides code and resources for exploring **annotator reliability** and **probabilistic consensus methods** in semantic segmentation for digital pathology.  
The framework integrates:
- Dataset preprocessing  
- Annotator self-consistency (SC) analysis  
- Training UNet models with SC-based labeling maps  

The project is structured into three main components:

- **Dataset preprocessing** – Scripts to process WSIs and compute patch-level statistics (e.g., saturation threshold).  
- **SC (Self-Consistency)** – Compute annotator self-consistency indices and generate SC-based probabilistic masks.  
- **UNet Training** – Train UNet models with SC and smooth labeling maps.  

---

## Requirements

You need **Python 3.9+**.  
Install dependencies with:

```bash
pip install -r requirements.txt
```



## Prerequisite: Data Preparation

You need to have extracted square regions from each WSI you intend to train on.<br>
To do so, you can take a look at [HS2P](https://github.com/clemsgrs/hs2p), which segments tissue and extract relevant patches at a given pixel spacing.


## Self-Consistency (SC)
We use three datasets: PANDA, Gleason, and BCSS (Breast).

For each dataset, a dedicated Jupyter notebook (SC_{dataset}.ipynb) is provided to:

- Analyze annotations

- Compute SC indices and weights

- Generate SC-based masks for training


## Unet training
Different configuration files (config.yaml) must be adapted for each dataset.

The main training script is: model-handler.py

Run training with:
```bash
python main.py -dc config.yaml -ef experiments/sc > experiments/sc/log.out
```


---
## Citation
If you use this code, please cite the preprint:

```bibtex
@article{galvezjimenez2024annotator,
  title={Annotator Reliability and Probabilistic Consensus for Semantic Segmentation in Digital Pathology},
  author={Galvez-Jimenez, Laura and ...},
  journal={SSRN Preprint},
  year={2024},
  url={https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5271896}
}
```