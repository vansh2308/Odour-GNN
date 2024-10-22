# OdourGNN
#### Deep Learning based molecule odour predictor given their chemical structure, using Graph Neural Networks

<img src="figures/readme-hero.png" width="100%">

### Features
- Dataset preparation
- ML pipeline for multiclass predictions (138 classes) 
- Molecular visualization
- GNN models - GIN, GAT, GraphSAGE
- Model training & inference
- Inference Explainations
- Relative atom-bond importance visualization
- Hyperarameter tuning 
- Embedding Space visualisation

### Results 
##### Hyperparameter Tuning 
<div align="center">
<img src="./figures/plots/lr_acc_dropout.png" width="45%"> &nbsp;
<img src="/figures/plots/imp.png" width="50%">
<img src="/figures/plots/hyt.png" width="90%">
</div>

##### Finetuning 
<div align="center">
<img src="./figures/plots/acc.png" width="48%"> &nbsp;
<img src="/figures/plots/valloss.png" width="48%">
</div>


### Setup Instructions
__Requirements__
```
python3 = 3.11.3
pip = 24.2
```


Run the following in terminal
```bash
git https://github.com/vansh2308/Odour-GNN.git
cd ./Odour-GNN
python3 -m venv venv 
source venv/bin/activate
pip3 install -r ./requirements.txt
```

Voila😉 Now experiment using `python3 ./main.py`


### Author
- Github - [vansh2308](https://github.com/vansh2308)
- Website - [Vansh Agarwal](https://portfolio-website-self-xi.vercel.app/)