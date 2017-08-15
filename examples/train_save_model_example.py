# This example demonstrates how one might train a model and then save
# it for loading and testing with later.


from wekapy import *

# CREATE NEW MODEL INSTANCE WITH A CLASSIFIER TYPE

model = Model(classifier_type = "bayes.BayesNet")

# CREATE TRAINING INSTANCES. LAST FEATURE IS THE PREDICTION OUTCOME

instance1 = Instance()
instance1.add_features([ Feature(name="num_milkshakes", value=46, possible_values="numeric"),
    Feature(name="is_sunny", value=True, possible_values="{False, True}"),
    Feature(name="boys_in_yard", value=True, possible_values="{False ,True}") ])

instance2 = Instance()
instance2.add_features([ Feature(name="num_milkshakes", value=2, possible_values="numeric"),
    Feature(name="is_sunny", value=False, possible_values="{False, True}"),
    Feature(name="boys_in_yard", value=False, possible_values="{False ,True}") ])

model.add_train_instance(instance1)
model.add_train_instance(instance2)


# FINALLY, TRAIN AND SAVE THE TRAINED MODEL TO FILE

model.train(folds=2, save_as="/path/to/model.model")
