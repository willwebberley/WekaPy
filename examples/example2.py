# This example demonstrates:
#   - creation of the Model object
#   - training a model using a pre-existing ARFF file
#   - testing the tesiing instances against the trained model using an ARFF file
#   - accessing the predictions

from wekapy import *

model = Model(classifier_type = "bayes.BayesNet")
model.train(training_file = "train.arff")
model.test(test_file = "test.arff")

for prediction in model.predictions:
    print prediction
