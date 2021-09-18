## Folder Structure
```
hparams-template/
├── data/
│   └── FashionMNIST/
├── logs/
│   └── hp_tunning/
│      └── exp_n/
├── model.py
├── functions.py
├── train.py
├── main.py
└── run.sh
```
- ```data/FashionMNIST/```: download=True
- ```logs/```: Tensorboard hparam logs
- ```run.sh```: Shell script
    - ```--EPOCH```: Training Epoch
    - ```--OPTIMIZER```: Optimizer
    - ```LR```: Learning rate
    - ```EXP_NUM```: Experiment count
    - ```BATCH_SIZE```: Batch size

## Output
![image](https://user-images.githubusercontent.com/85881032/133894950-aaec49c7-0c19-4c5a-b0a9-0f50f89e817d.png)