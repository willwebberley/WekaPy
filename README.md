WekaPy v1.0
=================

A simple Python module to provide a very basic wrapper for the Weka toolkit.

Please note this module is in very early stages of development and probably will not work in some cases.

**Requires:**
* Working Python installation
* weka.jar
    * Download the JAR file from [their website](http://www.cs.waikato.ac.nz/ml/weka/downloading.html)
    * Place the `weka.jar` file in your system's Java classpath or in your project's directory

**Current features:**
* Create a classifier instance
    * Tested with Bayesian Networks and Trees, but should be compatible with most classifier types
* Train the classifier with a training dataset
    * Currently relies on you providing an ARFF file containing the instances you with to train the model with
    * Will soon allow adding instances programatically in Python (no ARFF required)
* Produce predictions for a test dataset
    * Again, relies on an ARFF file containing the instances you wish to test on the trained model
    * Generates predictions which can then be retrieved from the Classifier object

Example usage
---------------

The module can be very easily used as shown in the following example. The functionality is built around the `Model` class, which requires a classifier type when constructed.

```python
from wekapy import *

model = Model(classifier_type = "trees.J48")
model.train(data_file="train.arff")
model.test(test_set = "test.arff")
```

The list of predictions can then be retrieved like so:
```python
predictions = model.predictions
```

Optional Arguments
---------------------

Currently, the `Model` class also supports the following additional arguments during construction:

* `verbose` (`True` by default)
    * set `verbose = False` to prevent the module from printing out status reports to STDOUT (fatal errors will still be printed)
* `max_memory` (`1500` by default)
    * change the value of `max_memory` to set the maximum allowable memory (in MB) for the Java Virtual Machine to run with Weka
    * You may need to increase this if you get stack overflow Exceptions
    * You may need to decrease this if your machine does not have enough RAM to support this

For example, to instantiate the `Model` object with additional arguments, you could use:

```python
model = Model(classifier_type = "bayes.BayesNet", verbose = False, max_memory = 1000)
```
