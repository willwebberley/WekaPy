import subprocess
import os
import time

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
        return_s = str(self.index)+":"
        return_s = return_s+"pred: "+str(self.predicted_value)+" with prob: "+str(self.probability)
        return return_s

class Instance:
    def __init__(self, feature_type = None, feature_value = None):
        self.feature_type = feature_type
        self.feature_value = feature_value
        
class Model:
    def __init__(self, classifier_type = None, max_memory = 1500, verbose = True):
        if classifier_type == None or not isinstance(classifier_type, str):
            print "Please provide a classifier type."
            return False
        if not isinstance(max_memory, int):
            print type(max_memory)
            print "'max_memory' argument must be of type (int)."
            return False
        self.classifier = classifier_type
        self.max_memory = max_memory
        self.training_instances = []
        self.predictions = []
        self.verbose = verbose
        self.trained = False

    # Train the model with the chosen classifier from features in an ARFF file
    def train(self, data_file = None, output = "my_model.model"):
        if self.verbose: print "Training your classifier..."
        start_time = time.time()
        if data_file == None or output == None:
            print "Please provide a filename for the data_file and output_model"
            return False
        self.model_file = output
        process = subprocess.Popen(["java", "-Xmx"+str(self.max_memory)+"M", "weka.classifiers."+self.classifier, "-t", data_file, "-d", output], stdout=subprocess.PIPE)
        process_output = process.communicate()[0]
        end_time = time.time()
        self.trained = True
        if self.verbose: print "Training complete (time taken = %.2fs)." % (end_time-start_time)

    # Generate predictions from the trained model from test features in an ARFF file
    def test(self, test_set = None):       
        if self.verbose: print "Generating predictions for your test set..."
        start_time = time.time()
        if not self.trained:
            print "The classifier has not yet been trained. Please call train() first"
            return False
        if test_set == None or self.model_file == None:
            print "Please provide a filename for the test_set and model"
            return False
        process = subprocess.Popen(["java", "-Xmx"+str(self.max_memory)+"M", "weka.classifiers."+self.classifier, "-T", test_set, "-l", self.model_file, "-p", "0"], stdout=subprocess.PIPE)
        output = process.communicate()[0]
        lines = output.split("\n")
        instance_predictions = []
        for line in lines:
            pred = line.split()
            if len(pred) >= 4 and pred[0].startswith("1"):
                index = int(pred[0])
                ob_cat = int((pred[1].split(":"))[0])
                ob_val = str((pred[1].split(":"))[1])
                p_cat = int((pred[2].split(":"))[0])
                p_val = str((pred[2].split(":"))[1])
                error = False
                prob = 0.0
                if "+" in pred[3]:
                    error = True
                    prob = float(pred[4])
                else:
                    prob = float(pred[3])
                prediction = Prediction(index, ob_cat, ob_val, p_cat, p_val, error, prob)
                instance_predictions.append(prediction)       
        self.predictions = instance_predictions
        end_time = time.time()
        if self.verbose: print "Testing complete (time taken = %.2fs)." % (end_time-start_time) 
        return instance_predictions
