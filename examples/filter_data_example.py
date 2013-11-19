# This example demonstrates:
#   - creation of the Model object
#   - creation of training instances of features
#   - training a model
#   - creation of testing instances of features
#   - testing the tesiing instances against the trained model
#   - accessing the predictions


from wekapy import *

# CREATE NEW MODEL INSTANCE WITH A CLASSIFIER TYPE

model = Model(classifier_type = "bayes.BayesNet")

# CREATE TRAINING INSTANCES. LAST FEATURE IS THE PREDICTION OUTCOME

instance1 = Instance()
instance1.add_feature(Feature(name="num_milkshakes",value=46,possible_values="real"))
instance1.add_feature(Feature(name="is_sunny",value=True,possible_values="{False, True}"))
instance1.add_feature(Feature(name="boys_in_yard",value=True,possible_values="{False ,True}"))

instance2 = Instance()
instance2.add_feature(Feature(name="num_milkshakes",value=2,possible_values="real"))
instance2.add_feature(Feature(name="is_sunny",value=False,possible_values="{False, True}"))
instance2.add_feature(Feature(name="boys_in_yard",value=False,possible_values="{False, True}"))

model.add_train_instance(instance1)
model.add_train_instance(instance2)

model.train(folds=2)


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

model.test()

# CHECK THE PREDICTIONS:
predictions = model.predictions
for prediction in predictions:
    print prediction
