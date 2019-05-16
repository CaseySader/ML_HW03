## Authors:
### Casey Sader & Lei Wang

## How to run the program:
### Your system needs to support python2.7 before compile

### command to generate run
* Run on ionosphere datatset:
```python
python NeuralNetwork.py -d ionosphere
```
example output:
`code()
Loading dataset
Feature Extraction
Beginning training
Epoch 1: training loss = 0.72561663507, test loss = 0.422335108695
Epoch 2: training loss = 0.681297607546, test loss = 0.554629393048
Epoch 3: training loss = 0.672293170259, test loss = 0.719649046129
Epoch 4: training loss = 0.686131813937, test loss = 0.858675360532
Epoch 5: training loss = 0.691383082703, test loss = 0.908847428365
Epoch 6: training loss = 0.673279233262, test loss = 0.842649187694
Epoch 7: training loss = 0.652690792309, test loss = 0.728942950953
Epoch 8: training loss = 0.644293871302, test loss = 0.612275911894
Epoch 9: training loss = 0.645926947761, test loss = 0.540099088035
Epoch 10: training loss = 0.643290293484, test loss = 0.520956286026
Training done.\n
Test Data Statistics:\n
              precision    recall  f1-score   support\n
           b       0.75      0.21      0.33        14
           g       0.89      0.99      0.94        92\n
   micro avg       0.89      0.89      0.89       106
   macro avg       0.82      0.60      0.64       106
weighted avg       0.87      0.89      0.86       106\n
`

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


## References:
* https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research
* https://www.openml.org/search?type=data
* [3Blue1Brown's YouTube series on Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)