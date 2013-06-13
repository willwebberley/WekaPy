from wekapy import *

# CREATE NEW MODEL INSTANCE WITH A CLASSIFIER TYPE

model = Model(classifier_type = "bayes.BayesNet")

# CREATE TRAINING INSTANCES

instance1 = Instance()
instance1.features.append(Feature(name="num_milkshakes",value=46,possible_values="real"))
instance1.features.append(Feature(name="is_sunny",value=True,possible_values="{False, True}"))
instance1.features.append(Feature(name="boys_in_yard",value=True,possible_values="{False ,True}"))

instance2 = Instance()
instance2.features.append(Feature(name="num_milkshakes",value=2,possible_values="real"))
instance2.features.append(Feature(name="is_sunny",value=False,possible_values="{False, True}"))
instance2.features.append(Feature(name="boys_in_yard",value=False,possible_values="{False, True}"))

instances = []
instances.append(instance1)
instances.append(instance2)

# CREATE TEST INSTANCES

test_instance1 = Instance()
test_instance1.features.append(Feature(name="num_milkshakes",value=44,possible_values="real"))
test_instance1.features.append(Feature(name="is_sunny",value=True,possible_values="{False, True}"))
test_instance1.features.append(Feature(name="boys_in_yard",value="?",possible_values="{False, True}"))

test_instance2 = Instance()
test_instance2.features.append(Feature(name="num_milkshakes",value=5,possible_values="real"))
test_instance2.features.append(Feature(name="is_sunny",value=False,possible_values="{False, True}"))
test_instance2.features.append(Feature(name="boys_in_yard",value="?",possible_values="{False, True}"))

test_instances = []
test_instances.append(test_instance1)
test_instances.append(test_instance2)

# TRAIN AND TEST MODEL
model.train(instances = instances, folds = 2)
model.test(instances = test_instances)

# CHECK THE PREDICTIONS:
predictions = model.predictions
for prediction in predictions:
    print prediction
