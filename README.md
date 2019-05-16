## Authors:
### Casey Sader & Lei Wang

## Datasets (python script downloads these for you):
* [ionosphere dataset](https://www.openml.org/d/59): Analyze radar data and predict if the data shows evidence of a structure in the ionosphere (good) or not (bad)
* [mnist dataset](https://www.openml.org/d/554): Analyze images of handwritten digits to create a model that can predict the digit

## Process:
1. Read in the command line arguments to determine the dataset and hyperparameters for the model.
2. Load the dataset and convert all labels to integers.
3. Create features array and labels array.
4. (if mnist, divide all values by 255 to normalize between 0 and 1) Then reshape the labels array so that each label is one-hot-encoded for the possible label values.
5. Create training and testing sets.
6. Initialize weights and biases using random values.
7. For each epoch and batch, perform forward propagation and backwards propagation using the sigmoid activation function to get the gradient values used to determine the new weights and biases. Update weights and biases then calculate loss value for both the training and testing sets. Move on to next epoch.
8. Use forward propagation on the testing features to make predictions and then calculate metrics using the true label values.

## How to run the program:
### Your system needs to support python2.7
* You may need to run
```python
pip install sklearn
pip install numpy
```
### Command to generate run
* #### Run on ionosphere datatset:
```python
python NeuralNetwork.py -d ionosphere
```
example output:

>Loading dataset  
>Feature Extraction  
>Beginning training  
>Epoch 1: training loss = 0.72561663507, test loss = 0.422335108695  
>Epoch 2: training loss = 0.681297607546, test loss = 0.554629393048  
>Epoch 3: training loss = 0.672293170259, test loss = 0.719649046129  
>Epoch 4: training loss = 0.686131813937, test loss = 0.858675360532  
>Epoch 5: training loss = 0.691383082703, test loss = 0.908847428365  
>Epoch 6: training loss = 0.673279233262, test loss = 0.842649187694  
>Epoch 7: training loss = 0.652690792309, test loss = 0.728942950953  
>Epoch 8: training loss = 0.644293871302, test loss = 0.612275911894  
>Epoch 9: training loss = 0.645926947761, test loss = 0.540099088035  
>Epoch 10: training loss = 0.643290293484, test loss = 0.520956286026  
>Training done.  
>  
>Test Data Statistics:  
>  
>              precision    recall  f1-score   support  
>  
>           b       0.75      0.21      0.33        14  
>           g       0.89      0.99      0.94        92  
>  
>   micro avg       0.89      0.89      0.89       106  
>   macro avg       0.82      0.60      0.64       106  
>weighted avg       0.87      0.89      0.86       106  
>  


* #### Run on mnist dataset:
```python
python NeuralNetwork.py -d mnist_784
```
example output:

>Loading dataset  
>Feature Extraction  
>Beginning training  
>Epoch 1: training loss = 0.695494875969, test loss = 0.669506317081  
>Epoch 2: training loss = 0.462445647993, test loss = 0.440131286278  
>Epoch 3: training loss = 0.386904474112, test loss = 0.368653975245  
>Epoch 4: training loss = 0.349383001737, test loss = 0.334040871395  
>Epoch 5: training loss = 0.324961520066, test loss = 0.312349764054  
>Epoch 6: training loss = 0.306472964474, test loss = 0.295567974386  
>Epoch 7: training loss = 0.292175051203, test loss = 0.283073804934  
>Epoch 8: training loss = 0.279983777634, test loss = 0.27239595451  
>Epoch 9: training loss = 0.269675758606, test loss = 0.263298758342  
>Epoch 10: training loss = 0.261015109066, test loss = 0.25573168386  
>Training done.  
>  
>Test Data Statistics:  
>  
>              precision    recall  f1-score   support  
>  
>           0       0.97      0.94      0.96      2144  
>           1       0.98      0.95      0.96      2359  
>           2       0.91      0.93      0.92      2073  
>           3       0.91      0.91      0.91      2137  
>           4       0.95      0.91      0.93      2120  
>           5       0.89      0.90      0.89      1884  
>           6       0.96      0.94      0.95      2061  
>           7       0.93      0.93      0.93      2218  
>           8       0.89      0.92      0.90      2005  
>           9       0.89      0.92      0.91      1999  
>  
>   micro avg       0.93      0.93      0.93     21000  
>   macro avg       0.93      0.93      0.93     21000  
>weighted avg       0.93      0.93      0.93     21000  
>  

* #### Run and specify hyperparameter values 
```python
python NeuralNetwork.py -d <dataset> -n <num_hidden_layer_nodes> -r <learning_rate> -e <epochs> -b <batch_size>
```

## References:
* https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research
* https://www.openml.org/search?type=data
* [3Blue1Brown's YouTube series on Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)