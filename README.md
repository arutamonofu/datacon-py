```
.
├── configs
│   └── main_config.yaml
├── data
│   ├── 01_raw
│   │   ├── chembl_CHEMBL4822_raw.csv
│   │   └── pubchem_P56817_raw.csv
│   ├── 02_processed
│   │   ├── all_processed.csv
│   │   ├── chembl_CHEMBL4822_processed.csv
│   │   └── pubchem_P56817_processed.csv
│   ├── 03_features
│   │   └── CHEMBL4822_rdkit.csv
│   ├── 04_generated
│   │   └── sampling.csv
│   └── 05_selected
│       ├── 01_program_filtered.csv
│       ├── 02_pic50_filtered.csv
│       ├── 03_for_swissadme.txt
│       ├── 04_final_ranked_hits.csv
│       └── swissadme.csv
├── main.py
├── models
│   ├── columns.json
│   └── model.joblib
├── requirements.txt
├── scripts
│   ├── generate_molecules.py
│   ├── prepare_data.py
│   ├── prepare_predictor.py
│   └── select_molecules.py
└── src
    ├── data
    │   ├── descriptors.py
    │   ├── download.py
    │   ├── preprocess.py
    │   └── score.py
    ├── predictor.py
    └── utils.py
```
