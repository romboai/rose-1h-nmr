# Third-party data attribution

ROSE code and pretrained weights are Apache-2.0 (`LICENSE`, `NOTICE`).
**This repository does not ship spectra.** The tables below list sources used
to *train* ROSE-Pretrain-L or to *evaluate* it. Licenses apply to those
upstream files, not to the ROSE checkpoint.

If you need the original data, download it from the listed URL and follow
that provider’s terms. Where a license is mixed or study-level, treat
redistribution as **not granted by ROSE**.

Paper: [ChemRxiv](https://doi.org/10.26434/chemrxiv.15007823/v1).
Catalog IDs match Supporting Information Table S2 (`tbl:si-source-splits`).

## Pretraining sources (ROSE-Pretrain-L)

| Source | Catalog ID | Access | License (provider) | Redistribute spectra? | Cite |
|--------|------------|--------|--------------------|-----------------------|------|
| NMRexp | `benchmark_nmrexp_17296666` | [Zenodo 17296666](https://doi.org/10.5281/zenodo.17296666) | CC BY 4.0 | Get from Zenodo; not in this repo | Wang et al., *Sci. Data* **12**, 1954 (2025). [10.1038/s41597-025-06245-5](https://doi.org/10.1038/s41597-025-06245-5) |
| IBM MSD | `zenodo_ibm_msd_14770232` | [Zenodo 14770232](https://doi.org/10.5281/zenodo.14770232) | CDLA-Sharing-1.0 | Get from Zenodo (share-alike on *data*); not in this repo | Alberts et al., NeurIPS 2024. [arXiv:2407.17492](https://arxiv.org/abs/2407.17492) |
| ChefNMR USPTO | `benchmark_chefnmr_uspto` | [Zenodo 17766755](https://doi.org/10.5281/zenodo.17766755) | CDLA-Sharing-1.0 (ChefNMR release) | Get from ChefNMR Zenodo; not in this repo | Xiong et al., NeurIPS 2025. [arXiv:2512.03127](https://arxiv.org/abs/2512.03127) |
| Sagmeister | `zenodo_sagmeister_6064586` | [Zenodo 6064586](https://doi.org/10.5281/zenodo.6064586) | CC BY 4.0 | Get from Zenodo; not in this repo | Sagmeister et al. (2022). [10.5281/zenodo.6064586](https://doi.org/10.5281/zenodo.6064586) |
| NMRBank train | `benchmark_nmrbank_train` | Original distributors (NMRExtractor) | See Chem. Sci. paper / authors | Not in this repo | Wang et al., *Chem. Sci.* **16**, 11548 (2025). [10.1039/D4SC08802F](https://doi.org/10.1039/D4SC08802F) |
| HMDB | `hmdb` | [hmdb.ca](https://hmdb.ca) | [CC BY-NC 4.0](https://hmdb.ca/about) | Academic/non-commercial data use; **commercial redistribution of HMDB data needs Wishart lab permission**. Not in this repo | Wishart et al., *Nucleic Acids Res.* **50**, D622 (2022). [10.1093/nar/gkab1062](https://doi.org/10.1093/nar/gkab1062) |
| NMR2Struct SpectraBase | `benchmark_nmr2struct_13892026` | [Zenodo 13892026](https://doi.org/10.5281/zenodo.13892026) | CC BY 4.0 on that deposit; SpectraBase is Wiley | Do **not** scrape Wiley SpectraBase. Use the authors’ Zenodo dump. Not in this repo | Hu et al., *ACS Cent. Sci.* **10**, 2162 (2024). [10.1021/acscentsci.4c01132](https://doi.org/10.1021/acscentsci.4c01132) |
| ChefNMR SpectraBase | `benchmark_chefnmr_spectrabase` | [Zenodo 17766755](https://doi.org/10.5281/zenodo.17766755) | CDLA-Sharing-1.0 (ChefNMR release); SpectraBase is Wiley | Same as NMR2Struct: do not scrape Wiley. Not in this repo | Xiong et al., NeurIPS 2025. [arXiv:2512.03127](https://arxiv.org/abs/2512.03127) |
| ChefNMR SpectraNP | `benchmark_chefnmr_spectranp` | [Zenodo 17766755](https://doi.org/10.5281/zenodo.17766755) | CDLA-Sharing-1.0 | Get from ChefNMR Zenodo; not in this repo | Xiong et al., NeurIPS 2025. [arXiv:2512.03127](https://arxiv.org/abs/2512.03127) |
| NMRShiftDB2 | `nmrshiftdb2` | [nmrshiftdb2](https://nmrshiftdb.nmr.uni-koeln.de/) | [nmrshiftdb2 Database License](https://nmrshiftdb.nmr.uni-koeln.de/nmrshiftdbhtml/nmrshiftdb2datalicense.txt) (ODbL-derived) | Get from NMRShiftDB2; share-alike on the database. Not in this repo | NMRShiftDB2 project |
| MetaboLights | `metabolights` | [EBI MetaboLights](https://www.ebi.ac.uk/metabolights/) | Study-level (CC0 from Apr 2025 submissions; older studies: EMBL-EBI terms) | Per-study; not in this repo | Yurekten et al., *Nucleic Acids Res.* **52**, D640 (2024). [10.1093/nar/gkad1045](https://doi.org/10.1093/nar/gkad1045) |
| GISSMO | `bmr_gissmo` | [BMRB / GISSMO](https://gissmo.nmrfam.wisc.edu/) | BMRB public metabolomics collection (free of charge; see [data policy](https://bmrb.io/metabolomics/data_policy.shtml)) | Get from BMRB; not in this repo | Dashti et al., *Anal. Chem.* **90**, 10646 (2018). [10.1021/acs.analchem.8b02660](https://doi.org/10.1021/acs.analchem.8b02660) |
| NMRNet train | `benchmark_nmrnet_train` | NMRNet authors / paper | See *Nat. Comput. Sci.* 2025 | Not in this repo | Xu et al., *Nat. Comput. Sci.* **5**, 292 (2025). [10.1038/s43588-025-00783-z](https://doi.org/10.1038/s43588-025-00783-z) |
| BMRB | `bmrb` | [bmrb.io](https://bmrb.io) | Public deposition; free access (see BMRB policy) | Get from BMRB; not in this repo | Ulrich et al., *Nucleic Acids Res.* **36**, D402 (2008). [10.1093/nar/gkm957](https://doi.org/10.1093/nar/gkm957) |
| nmrXiv | `nmrxiv` | [nmrxiv.org](https://nmrxiv.org) | Per-project (often CC BY 4.0; depositor chooses) | Per-project; not in this repo | nmrXiv |

## Evaluation resources (not used as pretrain)

These stay with their original distributors. ROSE ships **ID lists only** under `indices/` (InChIKey-14 holdout and split manifests), not spectra.

| Resource | Role | Access | License (provider) | Cite |
|----------|------|--------|--------------------|------|
| QIB edible oils (60 MHz) | Frozen-encoder classification | [QIBChemometrics/NMR_Spectra_Edible_Oils](https://github.com/QIBChemometrics/NMR_Spectra_Edible_Oils) | CC0-1.0 | Gunning et al., *Food Chem.* **370**, 131028 (2022). [10.1016/j.foodchem.2021.131028](https://doi.org/10.1016/j.foodchem.2021.131028) |
| NMRNet | Structure→shift test pickle | NMRNet release | See Xu et al. 2025 | Xu et al., *Nat. Comput. Sci.* **5**, 292 (2025). [10.1038/s43588-025-00783-z](https://doi.org/10.1038/s43588-025-00783-z) |
| NMR-Solver / Zenodo | Retrieval eval | [Zenodo 16952024](https://doi.org/10.5281/zenodo.16952024) | CC BY 4.0 | Jin et al. [arXiv:2509.00640](https://arxiv.org/abs/2509.00640); [10.5281/zenodo.16952024](https://doi.org/10.5281/zenodo.16952024) |
| NMRGym | Retrieval eval | [huggingface.co/datasets/meaw0415/NMRGym](https://huggingface.co/datasets/meaw0415/NMRGym) | See dataset card | Fang et al. [arXiv:2601.15763](https://arxiv.org/abs/2601.15763) |
| NMRBank | Retrieval eval | Original distributors | See Wang et al. 2025 | Wang et al., *Chem. Sci.* **16**, 11548 (2025). [10.1039/D4SC08802F](https://doi.org/10.1039/D4SC08802F) |
| NMRformer | Peak-detection eval | Authors / paper SI | See Zhou et al. 2025 | Zhou et al., *Anal. Chem.* **97**, 904 (2025). [10.1021/acs.analchem.4c05632](https://doi.org/10.1021/acs.analchem.4c05632) |

Release policy: **recipes and ID manifests only**. No ROSE-Pretrain-S/L parquet on GitHub, Hugging Face, or Zenodo.

## What ROSE *does* ship

| Artifact | License | Where |
|----------|---------|--------|
| Inference code | Apache-2.0 | this repo |
| Pretrained weights | Apache-2.0 | [huggingface.co/romboai/rose-1h-nmr](https://huggingface.co/romboai/rose-1h-nmr) |
| Split / holdout **IDs** (no spectra) | Apache-2.0 | `indices/` |

Licenses above were read from provider pages / Zenodo metadata (2026-08).
They can change; the provider’s current terms win.
