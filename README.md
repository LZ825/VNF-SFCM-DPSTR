# Service Function Chain Live Migration with Bi-GRU Resource Prediction in Substrate Networks

The repository maintains the source codes of VNF-SFCM-DPSTR, a dynamic service function chain migration method based on network resource prediction. This method uses an online-trained Bi-GRU model to predict resource demands, identify nodes and links that are about to be overloaded, and employs an improved discrete particle swarm optimization algorithm to select the optimal migration target nodes, achieving seamless dynamic migration of the service function chain.

## Install

Install dependencies:

```
pip install -r requirements.txt
```

## RUN

Bi-GRU training

```
python Univariate_Bi-GRU.py
```

run migration simulation

```
python gnt_DPSO.py
```
