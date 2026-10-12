# ESA-CCI
Exploring European Space Agency's (ESA) Climate Change Initiative (CCI) datasets

This repo explores the Permafrost, Snow and Land Cover components of the ESA CCI Climate Research Data Package v4
(CRDP v4). The domain is the northern hemisphere pan-Arctic; the exploratory focus is Alaska, 2000–2020.

- [`pipeline/`](pipeline/): reproducible code to crawl the CEDA archive, download products and subset them to Alaska
- [`eda/`](eda/): exploratory analysis. Start with [`eda/EDA.md`](eda/EDA.md); sample plots are in [`eda/plots/`](eda/plots/)

Current status: Permafrost_cci v5.0 (NH, 1997–2023) is downloaded and subset. Snow_cci v4.0 and Land_Cover_cci are
catalogued only (listings and file headers).

## Quick start

```bash
micromamba create -f environment.yml && micromamba activate esa-cci
python pipeline/download.py --theme permafrost --dry-run   # show what would be downloaded
python pipeline/download.py --theme permafrost             # ~10 GB to data_root (pipeline/config.yaml)
python -I pipeline/subset.py --theme permafrost            # Alaska 2000–2020, native grid
python -I eda/inventory.py && python -I eda/plot_permafrost.py
```

Data live outside the repo, under `data_root` in [`pipeline/config.yaml`](pipeline/config.yaml)
(currently `/import/beegfs/CMIP6/jdpaul3/esa-cci`).
