from wekapy import *

model = Model(classifier_type = "bayes.BayesNet")
model.train(training_file = "train.arff")
model.test(test_file = "test.arff")

for prediction in model.predictions:
    print prediction
