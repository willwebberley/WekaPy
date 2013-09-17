WekaPy v1.2
=================

A simple Python module to provide a wrapper for some of the basic functionality of the Weka toolkit. The project focuses on the *classification* side of Weka, and does not consider clustering, distributions or any visualisation functions at this stage.

Weka is a machine learning tool, allowing you to classify data based on a set of its attributes and for generating predictions for unseen feature instances.

This module abstracts the use of ARFF files, making Weka much easier to use programmatically in Python. 

Please note that this project is in very early stages of development and probably will not work in some cases.

**Prerequisites:**
* wekapy.py (from this repo)
* Working Python installation
* Working Java installation
* weka.jar
    * Download the JAR file from [Weka's website](http://www.cs.waikato.ac.nz/ml/weka/downloading.html)
    * Place the `weka.jar` file in your system's Java classpath or in your project's directory (or, at least, somewhere you can import it into your project from)

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
* Save / load models
    * Save a model trained using WekaPy
    * Load a pre-trained model (from Weka, previous uses of WekaPy, etc.)
* Exporting data to ARFF format
    * WekaPy can generate ARFF files for your training and/or test data
    * Useful on its own if you'd rather use the GUI for making classifications

Example usage
---------------

Please see the `examples/` directory for full example uses. These include:
* Programmatically training and testing a model (`example1.py`)
* Training and testing a model using ARFF files (`example2.py`)
* Training and saving a model to file for testing with later (`train_save_model_example.py`)
* Loading a trained model from file and testing against it (`load_test_model_example.py`)

For more detailed documentation, please read on.

The module can be very easily used as shown in the following example. The functionality is built around the `Model` class, which requires a classifier type when constructed.

```python
from wekapy import *

model = Model(classifier_type = "trees.J48")
model.train(training_file = "train.arff")
model.test(test_file = "test.arff")
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

1. `Model` construction
-----------------------

**1.1 Standard construction**

Construct the `Model` object very simply:
```python
model = Model(classifier_type = "trees.J48")
```

**1.2 Optional arguments**

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

2. Training the model
----------------------

Currently, there are two methods for training the model - either through ARFF files or by providing a list of instances.
Both require the `Model` object to be instantiated first (see above).

When training the model, any models trained previously with this `Model` object will be replaced by the new model.

You will need to provide either an ARFF file or a list of Instances in order for the train to be successful.

**2.1 Using ARFF Files**

Generate an ARFF file and pass this to the `train()` function:
```python
model.train(training_file = "train.arff")
```

**2.2 Using the Instance object**

If you would rather carry this out programmatically, then you can instead provide a list of Instance objects. 

An Instance simply contains a list of Features, and can be instantiated as follows:
```python
instance1 = Instance()

feature1 = Feature(name="num_milkshakes",value=46,possible_values="real")
feature2 = Feature(name="is_sunny",value=True,possible_values="{False, True}")
feature3 = Feature(name="boys_in_yard",value=True,possible_values="{False ,True}")
 
instance1.add_feature(feature1)
instance1.add_feature(feature2)
instance1.add_feature(feature3)

instance2  = Instance()
...
```

The final feature in each instance will be the that predictions are made against (the 'class').

When you've created all of your training instances, add them to your untrained model:
```python
model.add_train_instance(instance1)
model.add_train_instance(instance2)
...
```

Finally, train the model:
```python
model.train()
```

In the background, WekaPy generates an ARFF file and saves this and the trained model in its data directories. If desired, these can be found using:
```python
training_arff = model.training_file
trained_model = model.model_file
```

** 2.3 Optional arguments**

To configure the training more precisely, you can also set a different directory for the model and specify the number of cross-validation folds the training algorithm will carry out:
* `save_as` (hidden, by default, to improve seamlessness of use)
    * Set `save_as = "path/to/model"` to save the model in a different directory and with your own name.
    * This saved model can then be used later by passing it to the `test()` method as described later.
* `folds` (`10` by default)
    * Set `folds = x` to specify the number of cross-validation folds you want the algorithm to carry out.
    * If your Instance list is short, you may need to reduce this.
* `instances`
    * Pass a list of instances to `train()` instead of using `add_train_instance()`, if desired.
* `training_file`
    * Pass a training ARFF file to `train()` instead of programmatically adding features. This method is covered in section 2.1.

3. Testing with the trained model
--------------------------------

As with the training, there are two methods for testing with the model. You must provide either an ARFF file or a list of Instances for the testing to be successful.

**3.1 Using ARFF files**

Test using your own ARFF file as follows:
```python
model.test(test_file = "test.arff")
```

**3.2 Using a list of Instances**

Generate a list of Instances as described earlier. When testing, if the outcome feature is unknown, then use a `"?"` to signify this. For example:
```python
test_instance1 = Instance()

test_feature1 = Feature(name="num_milkshakes",value=5,possible_values="real")
test_feature2 = Feature(name="is_sunny",value=False,possible_values="{False, True}")
test_feature3 = Feature(name="boys_in_yard",value="?",possible_values="{False, True}")

test_instance1.add_feature(test_feature1)
test_instance1.add_feature(test_feature2)
test_instance1.add_feature(test_feature3)

test_instance2 = Instance()
...
```

Now add the testing instances to the model and test the,=m:
```python
model.add_test_instance(test_instance1)
model.add_test_instance(test_instance2)
...

model.test()
```

As before, an ARFF file is generated and this is used to test against the model.


**3.3 Optional arguments**

You can specify the use of a different model for testing against, and thus skip out the `train()` section, if you desire. This could be useful if you have already used `train()` and chose to save the model elsewhere, you have trained the model using Weka's GUI, using someone else's model, etc.
* `model_file` (`None` by default)
    * Set `model_file = "path/to/model.model"` to test with this model instead. 
    * Any models trained previously will be discarded by the current `Model` object and replaced by this one.
* `instances`
    * Pass a list of Instances to `test()` instead of using the `add_test_instance()` method demonstrated in 3.2.
* `test_file`
    * Pass a test file to `test()` as demonstrated in 3.1.


4 Accessing the predictions
--------------------------

If the testing is successful, a list of Predictions will be generated, containing a Prediction object for each Instance in the test ARFF file or the list of test Instances.

Below is an example demonstrating how to access the Predictions:
```python
predictions = model.predictions
for prediction in predictions:
    print prediction
```

**4.1 Further information**

For each Prediction object, these fields are available:
* `index` - integer representing the number of that Prediction. This equates to that Instance in the test Instance set or ARFF file. For example, the Prediction with `index = 1` is the prediction for the *first* instance in the test set.
* `observed_category` - integer representing the category number of the observed value (will not be available if observed value is unknown)
* `observed_value` - the observed outcome feature for this Instance
* `predicted_category` - integer representing the category number of the predicted value
* `predicted_value` - the predicted outcome feature for this Instance
* `error` - this will be `True` if the predicted value differs from the observed value. Therefore, this will be unavaialble if the observed value is unknown.
* `probability` - the probability with which the classifier believes the predicted value to be correct.
