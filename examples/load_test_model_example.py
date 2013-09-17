# This example demonstrates loading a pre-existing trained model and using
# this to test against.

from wekapy import *

# CREATE NEW MODEL INSTANCE WITH A CLASSIFIER TYPE

model = Model(classifier_type = "bayes.BayesNet")


# LOAD A PREVIOUSLY TRAINED MODEL INTO OUR model OBJECT FOR TESTING AGAINST

model.load_model("/path/to/model.model")


# CREATE TEST INSTANCES

test_instance1 = Instance()
test_instance1.add_feature(Feature(name="num_milkshakes",value=44,possible_values="real"))
test_instance1.add_feature(Feature(name="is_sunny",value=True,possible_values="{False, True}"))
test_instance1.add_feature(Feature(name="boys_in_yard",value="?",possible_values="{False, True}"))

test_instance2 = Instance()
test_instance2.add_feature(Feature(name="num_milkshakes",value=5,possible_values="real"))
test_instance2.add_feature(Feature(name="is_sunny",value=False,possible_values="{False, True}"))
test_instance2.add_feature(Feature(name="boys_in_yard",value="?",possible_values="{False, True}"))

model.add_test_instance(test_instance1)
model.add_test_instance(test_instance2)


# FINALLY, TEST AGAINST THE LOADED MODEL

model.test()

# CHECK THE PREDICTIONS:
predictions = model.predictions
for prediction in predictions:
    print prediction
