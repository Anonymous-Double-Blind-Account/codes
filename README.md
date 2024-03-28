# codes

## Simple Start
### MNIST with 9 silos
```shell
python train.py --data mnist --num_silo 9 --num_dist 3 --sample 2000 --encoder cnn --depth 3 --num_codes 64 --seg 1 --round 20 --epoch 20 --step 20 --thd 0.1 --workdir /your/save/folder
```

### MNIST with 5 silos (unbalanced)
```shell
python train.py --data mnist --num_silo 5 --num_dist 3 --sample 2000 --encoder cnn --depth 3 --num_codes 64 --seg 1 --round 20 --round_plus 5 --epoch 20 --step 20 --thd 0.3 --workdir /your/save/folder
```

### FMNIST with 9 silos
```shell
python train.py --data fmnist --num_silo 9 --num_dist 3 --sample 2000 --encoder cnn --depth 3 --num_codes 128 --seg 1 --round 30 --epoch 20 --step 20 --thd 0.1 --workdir /your/save/folder
```
