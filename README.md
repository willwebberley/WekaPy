WekaPy v1.1
=================

A simple Python module to provide a wrapper for some of the basic functionality of the Weka toolkit.

Please note this module is in very early stages of development and probably will not work in some cases.

**Prerequisites:**
* weakpy.py (from this repo)
* Working Python installation
* Working Java installation
* weka.jar
    * Download the JAR file from [their website](http://www.cs.waikato.ac.nz/ml/weka/downloading.html)
    * Place the `weka.jar` file in your system's Java classpath or in your project's directory

**Current features:**
* Create a classifier instance
    * Tested with Bayesian Networks and Trees, but should be compatible with most classifier types
* Train the classifier with a training dataset
    * Provide an ARFF file containing the instances you with to train the model with
    * Or, provide a list of Instance objects
* Produce predictions for a test dataset
    * Provide an ARFF file containing the instances you wish to test on the trained model
    * Or, provide a list of Instance objects
    * Generates a list of Prediction objects which can then be retrieved from the Model object

Example usage
---------------

The module can be very easily used as shown in the following example. The functionality is built around the `Model` class, which requires a classifier type when constructed.

```python
from wekapy import *

model = Model(classifier_type = "trees.J48")
model.train(data_file = "train.arff")
model.test(test_set = "test.arff")
```

The list of predictions can then be retrieved like so:
```python
prediction = model.predictions[0]
observed_value = prediction.observed_value
predicted_value = prediction.predicted_value
probability = prediction.probability
```


More detailed documentation
=========================

`Model` construction
-----------------------

**Standard construction**

Construct the `Model` object very simply:
```python
model = Model(classifier_type = "trees.J48")
```

**Optional arguments**

Currently, the `Model` class also supports the following additional arguments during construction:

* `verbose` (`True` by default)
    * set `verbose = False` to prevent the module from printing out status reports to STDOUT (fatal errors will still be printed)
* `max_memory` (`1500` by default)
    * change the value of `max_memory` to set the maximum allowable memory (in MB) for the Java Virtual Machine to run with Weka
    * The `Model` itself does not use this memory - only Weka when it is training and testing
    * You may need to increase this if you get stack overflow Exceptions
    * You may need to decrease this if your machine does not have enough RAM to support this

For example, to instantiate the `Model` object with additional arguments, you could use:

```python
model = Model(classifier_type = "bayes.BayesNet", verbose = False, max_memory = 1000)
```

Training the model
----------------------

Currently, there are two methods for training the model - either through ARFF files or by providing a list of instances.
Both require the `Model` object to be instantiated first (see above).

You will need to provide either an ARFF file or a list of Instances in order for the train to be successful.

**Using ARFF Files**

Generate an ARFF file and pass this to the `train()` function:
```python
model.train(training_file = "train.arff")
```

**Using the Instance object**

If you would rather carry this out programmatically, then you can instead provide a list of Instance objects. 

An Instance simply contains a list of Features, which can be instantiated as follows:
```python
feature1 = Feature(name="num_milkshakes",value=46,possible_values="real")
feature2 = Feature(name="is_sunny",value=True,possible_values="{False, True}")
feature3 = Feature(name="boys_in_yard",value=True,possible_values="{False ,True}") 
```

Next, create an Instance object and append your features. The final feature will be the that predictions are made against:
```python
instance1 = Instance([feature1, feature2, feature3, ...])
```

Finally, pass a list of Instances to the `train()` method:
```python
model.train(instances = [instance1, instance2, ...])
```

In the background, WekaPy generates an ARFF file and saves this and the trained model in its data directories. These can be found using:
```python
training_arff = model.training_file
trained_model = model.model_file
```

**Optional arguments**

To configure the training more precisely, you can also set a different directory for the model and specify the number of cross-validation folds the training algorithm will carry out:
* `save_as`
    * Set `save_as = "path/to/model"` to save the model in a different directory and with your own name
    * This saved model can then be used later by passing it to the `test()` method as described later.
* `folds` (10 by default)
    * Set `folds = x` to specify the number of cross-validation folds you want the algorithm to carry out.
    * If your Instance list is short, you may need to reduce this.


Testing with the trained model
--------------------------------
