## Goal: Predict teeth roots

### Setup dev environment

```
conda env create -f environment.yaml
conda activate ml2txparser
pip install -r requirements-dev.txt
pip install -r baseline/requirements.txt
```

## Some commands

```
# python dataset_baseline.py
python train_baseline.py --data-file ./baseline/data/all.csv --iterations 500 --rebuild-split

```
