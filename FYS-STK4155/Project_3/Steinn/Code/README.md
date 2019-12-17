## XGBoost Balanced EMNIST analysis folder
There are three arguments to be passed into the main function:
* *-cm*, 'Create Model'. Use this if you would like to use the training data to generate models.
* *-gs*, 'Grid Search'. Use this to conduct a grid search. The grid search specifications can be tuned in the datasearch modules.
* *-em*, 'Evaluate Model'. Use this to evaluate the model by producing predictions using the testing data.
An example of how a grid search can be conducted, creating all the models and then evaluating them:
```
$ python3 main.py -cm -gs -em
``` 
If the models are already saved, you can simply evaluate them using:
```
$ python3 main.py -gs -em
```
