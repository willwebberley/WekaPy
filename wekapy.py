# Portions of this software Copyright 2013 Will Webberley
# Portions of this software Copyright 2013 Martin Chorley
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# The full license text is available at <http://www.gnu.org/licenses/>.


import subprocess
import os
import time
import uuid
import random

def run_process(options):
    start_time = time.time()
    process = subprocess.Popen(options, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process_output, process_error = process.communicate()
    if "Exception" in process_error:
        for line in process_error.split("\n"):
            if "Exception" in line:
                raise WekapyException(line.split(' ',1)[1])
    end_time = time.time()
    return process_output, end_time-start_time 

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
        return_s = str(self.index)+":\t"
        return_s = return_s+"observed: "+str(self.observed_value)+"\tpredicted: "+str(self.predicted_value)+"\tprob: "+str(self.probability)
        return return_s


# Feature class
#
# Used internally and externally to represent a feature of data.
# Each feature should contain a name and a value (for example, name = 'daylight_hours', value = 10)
# possible_values should be represented by a String type object indicating the possible feature values
# e.g. real, {true, false}, {0,1,2}, {tom, dick, harry}, etc.
class Feature:
    def __init__(self, name = None, value = None, possible_values=None):
        self.name = name
        self.value = value
        self.possible_values = possible_values
        

# Instance class
#
# Used internally and externally to represent a set of Feature objects.
# Essentially, an Instance object just maintains a list of Features.
class Instance:
    def __init__(self, features = None):
        self.features = features
        if features == None:
            self.features = []
    def add_feature(self, feature):
        if isinstance(feature, Feature):
            self.features.append(feature) 
        else:
            raise WekapyException("Argument 'feature' must be of type Feature.")

# Filter class
#
# Used to filter/preprocess data using one of the weka.filters classes. 
class Filter:
    def __init__(self, max_memory=1500, verbose=True):
        if not isinstance(max_memory, int):
            raise WekapyException("'max_memory' argument must be of type (int).")
            return False
        self.max_memory = max_memory
        self.id = uuid.uuid4()
        self.verbose = verbose

    def filter(self, filter_options=None, input_file_name=None, output_file=None, class_column="last"):
        if filter_options is None:
            raise WekapyException("A filter type is required")
            return False
        if input_file_name is None:
            raise WekapyException("An input file is needed for filtering")
            return False
        if output_file is None:
            output_file = "%s-filtered.arff" % (input_file_name.rstrip(".arff"))
        if self.verbose: print "Filtering input data..."
        options = ["java", "-Xmx"+str(self.max_memory)+"M"]
        options.extend(filter_options)
        options.extend(["-i", input_file_name, "-o", output_file, "-c", class_column])
        print options
        process_output, run_time = run_process(options)
        if self.verbose: print "Filtering complete (time taken = %.2fs)." % (run_time)
        return output_file

    def split(self, input_file_name=None, training_percentage=67, randomise=True, seed=None):
        if input_file_name is None:
            raise WekapyException("An input file is needed for filtering")
            return False
        if not isinstance(training_percentage, int):
            raise WekapyException("'training_percentage' argument must be of type (int).")
            return False
        if randomise == True and seed is None:
            seed = random.randint(0,1000)
        if randomise == True:
            if self.verbose: print "Randomising data order..."
            output_file = "%s-randomised.arff" % (input_file_name.rstrip(".arff"))
            process_output, run_time = run_process(["java", "-Xmx"+str(self.max_memory)+"M", "weka.filters.unsupervised.instance.Randomize", "-S", str(seed), "-i", input_file_name, "-o", output_file])
            input_file_name = output_file
            if self.verbose: print "Randomisation complete (time taken = %.2fs)." % (run_time)
        if self.verbose: print "Beginning split...\nCreating training set..."
        output_file = "%s-training.arff" % (input_file_name.rstrip(".arff"))
        process_output, run_time_training = run_process(["java", "-Xmx"+str(self.max_memory)+"M", "weka.filters.unsupervised.instance.RemovePercentage", "-P", str(training_percentage), "-V", "-i", input_file_name, "-o", output_file])
        if self.verbose: print "Creating testing set..."
        output_file = "%s-testing.arff" % (input_file_name.rstrip(".arff"))
        process_output, run_time_testing = run_process(["java", "-Xmx"+str(self.max_memory)+"M", "weka.filters.unsupervised.instance.RemovePercentage", "-P", str(training_percentage), "-i", input_file_name, "-o", output_file])    
        if self.verbose: print "Split complete (time taken = %.2fs)." % (run_time_training+run_time_testing)



# Model class
#
# Used externally, and is the main class for use with this library.
# The Model class should be instantiated as the first stage, from which it can be trained 
# and/or tested.
# Instantiate with a classifier_type (and any optional arguments)
class Model:
    def __init__(self, classifier_type = None, max_memory = 1500, verbose = True):
        if classifier_type == None or not isinstance(classifier_type, str):
            raise WekapyException("A classifier type is required for construction.")
            return False
        if not isinstance(max_memory, int):
            raise WekapyException("'max_memory' argument must be of type (int).")
            return False
        self.id = uuid.uuid4()
        self.model_dir = "wekapy_data/models"
        self.arff_dir = "wekapy_data/arff"
        self.classifier = classifier_type
        self.max_memory = max_memory
        self.training_instances = []
        self.testing_instances = []
        self.predictions = []
        self.verbose = verbose
        self.trained = False
        if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
        if not os.path.exists(self.arff_dir):
                os.makedirs(self.arff_dir)
    
    # Generate an ARFF file from a list of instances
    def create_ARFF(self, instances, type):
        output_arff = open(os.path.join(self.arff_dir, "%s-%s.arff" %(str(self.id), type)), "w")
        output_arff.write("@relation "+str(self.id)+"\n")
        for i, instance in enumerate(instances):
            if i == 0:
                for feature in instance.features:
                    output_arff.write("\t@attribute "+feature.name+" "+str(feature.possible_values)+"\n")
                output_arff.write("\n@data\n")
            strToWrite = ""
            for j, feature in enumerate(instance.features):
                if j == 0:
                    strToWrite = strToWrite + str(feature.value)
                else:
                    strToWrite = strToWrite + "," + str(feature.value)           
            output_arff.write(strToWrite+"\n")
        output_arff.close()
        if type == "training":
            self.training_file = self.arff_dir+"/"+str(self.id)+"-"+type+".arff"
        if type == "test":
            self.test_file = self.arff_dir+"/"+str(self.id)+"-"+type+".arff"

    # Load a model, if it exists, and set this as the currently trained model for this
    # Model instance.
    def load_model(self, model_file):
        if os.path.exists(model_file):
            self.model_file = model_file
            self.trained = True
        else:
            raise WekapyException("Your model could not be found.")

    # Add a training instance to the model.
    def add_train_instance(self, instance):
        if isinstance(instance, Instance):
            self.training_instances.append(instance) 
        else:
            raise WekapyException("Argument 'instance' must be of type Instance.")

    # Add a testing instance to the model.
    def add_test_instance(self, instance):
        if isinstance(instance, Instance):
            self.testing_instances.append(instance) 
        else:
            raise WekapyException("Argument 'instance' must be of type Instance.")

    # Train the model with the chosen classifier from features in an ARFF file
    def train(self, training_file = None, instances = None, save_as = None, folds = 10):
        if self.verbose: print "Training your classifier..."
        if save_as == None:
            save_as = self.model_dir+"/"+str(self.id)+".model"
        if len(self.training_instances) == 0: # if add_train_instance not called:
            if training_file == None and instances == None:
                raise WekapyException("Please provide some train instances either by naming an ARFF train_set, providing a list of Instances, or calling add_train_instance().")
            if training_file == None:
                self.create_ARFF(instances, "training")
            if instances == None:
                self.training_file = training_file
        if len(self.training_instances) > 0: # if add_train_instance called:
            if training_file == None and instances == None:
                self.create_ARFF(self.training_instances, "training")
            # Prioritise adding fetures passed at calltime
            if training_file == None and instances is not None:
                self.create_ARFF(instances, "training")
            # Prioritise ARFF file passed at calltime
            if instances == None and training_file is not None:
                self.training_file = training_file

        self.model_file = save_as
        process_output, run_time = run_process(["java", "-Xmx"+str(self.max_memory)+"M", "weka.classifiers."+self.classifier, "-x", str(folds),"-t", self.training_file, "-d", save_as])
        self.trained = True
        if self.verbose: print "Training complete (time taken = %.2fs)." % (run_time)

    # Generate predictions from the trained model from test features in an ARFF file
    def test(self, test_file = None, instances = None, model_file = None):       
        if self.verbose: print "Generating predictions for your test set..."
        start_time = time.time()
        if not model_file == None:
            self.load_model(model_file)
        if not self.trained:
            raise WekapyException("The classifier has not yet been trained. Please call train() first")
        if len(self.testing_instances) == 0:
            if test_file == None and instances == None:
                raise WekapyException("Please provide some test instances either by naming an ARFF test_set, providing a list of Instances, or calling add_test_instance().")
            if test_file == None:
                self.create_ARFF(instances, "test")
            if instances == None:
                self.test_file = test_file
        if len(self.testing_instances) > 0:
            if test_file == None and instances == None:
                self.create_ARFF(self.testing_instances, "test")
            if test_file == None and instances is not None:
                self.create_ARFF(instances, "test")
            if instances == None and test_file is not None:
                self.test_file = test_file

        process_output, run_time = run_process(["java", "-Xmx"+str(self.max_memory)+"M", "weka.classifiers."+self.classifier, "-T", self.test_file, "-l", self.model_file, "-p", "0"])

        lines = process_output.split("\n")
        instance_predictions = []
        for line in lines:
            pred = line.split()
            if len(pred) >= 4 and not pred[0].startswith("=") and not pred[0].startswith("inst"):
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


class WekapyException(Exception):
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message
