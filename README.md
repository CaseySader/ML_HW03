## Authors:
### Casey Sader & Lei Wang

## How to run the program:
### Your system needs to support python2.7 before compile

### command to generate run
* Run on ionosphere datatset:
```python
python NeuralNetwork.py -d ionosphere
```
* Run on mnist dataset:
```python
python NeuralNetwork.py -d mnist_784
```
* Run and specify hyperparameter values 
```python
python NeuralNetwork.py -d <dataset> -n <num_hidden_layer_nodes> -r <learning_rate> -e <epochs> -b <batch_size>
```

## Datasets (python script downloads these for you):
* [ionosphere dataset](https://www.openml.org/d/59)
* [mnist dataset](https://www.openml.org/d/554)

## Processes:


## Reference:
* https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research
* https://www.openml.org/search?type=data
* [3Blue1Brown's YouTube series on Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)