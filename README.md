# SMC: Semiparametric Memory Conslodation

### Prepare the dataset:
Download the ImageNet-100 dataset, extract the dataset to any directory, and update the dataset path in ./utils/data.py.


### Create the environment:
```bash
conda env create -f environment.yml
```

### Enter the environment:
```bash
conda activate SMC
```

### Train the semiparametric memory first:
```bash
python train_sm.py
```

### Then train the SMC model:
```bash
python main.py -config ./exps/imagenet100/smc.json
```