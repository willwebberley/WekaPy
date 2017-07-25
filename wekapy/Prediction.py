# Prediction class
#
# Used internally and externally to WekaPy to represent a Prediction made as
# a result of running test data through a trained classifier.
# Each prediction effectively represents the classification of a set of instances.


class Prediction:
    def __init__(self, index, observed_1, observed_2, pred_1, pred_2, error, prob):
        self.index = int(index)
        self.observed_category = int(observed_1)
        self.observed_value = observed_2
        self.predicted_category = int(pred_1)
        self.predicted_value = pred_2
        self.error = bool(error)
        self.probability = float(prob)

    def __str__(self):
        return "{}:\tobserved: {}\tpredicted: {}\tprob: {}".format(str(self.index), str(self.observed_value),
                                                                   str(self.predicted_value), str(self.probability))
