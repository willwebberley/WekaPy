# Portions of this software Copyright 2017 Faiz Siddiqui
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


def decode_data(data):
    return data.decode('utf-8').strip()


def run_process(options):
    start_time = time.time()
    process = subprocess.Popen(options, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process_output, process_error = map(decode_data, process.communicate())
    if any(word in process_error for word in ["Exception", "Error"]):
        for line in process_error.split("\n"):
            if any(word in line for word in ["Exception", "Error"]):
                raise WekapyException(line.split(' ', 1)[1])
    end_time = time.time()
    return process_output, end_time - start_time


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


# Feature class
#
# Used internally and externally to represent a feature of data.
# Each feature should contain a name and a value (for example, name = 'daylight_hours', value = 10)
# possible_values should be represented by a String type object indicating the possible feature values
# e.g. numeric, <nominal-specification>, string, date [<date-format>] etc.
class Feature:
    def __init__(self, name=None, value=None, possible_values=None):
        self.name = name
        self.value = value
        self.possible_values = possible_values


# Instance class
#
# Used internally and externally to represent a set of Feature objects.
# Essentially, an Instance object just maintains a list of Features.
class Instance:
    def __init__(self, features=None):
        self.features = features
        if features is None:
            self.features = []

    def add_feature(self, feature):
        if isinstance(feature, Feature):
            self.features.append(feature)
        else:
            raise WekapyException("Argument 'feature' must be of type Feature.")

    def add_features(self, features_list):
        for feature in features_list:
            if isinstance(feature, Feature):
                self.features.append(feature)
            else:
                raise WekapyException("Argument 'feature' must be of type Feature.")


# Filter class
#
# Used to filter/pre-process data using one of the weka.filters classes.
class Filter:
    def __init__(self, max_memory=1500, classpath=None, verbose=False):
        if not isinstance(max_memory, int):
            raise WekapyException("'max_memory' argument must be of type (int).")
        self.classpath = classpath
        self.max_memory = max_memory
        self.id = uuid.uuid4()
        self.verbose = verbose

    def filter(self, filter_options=None, input_file_name=None, output_file=None, class_column="last"):
        if filter_options is None:
            raise WekapyException("A filter type is required")
        if input_file_name is None:
            raise WekapyException("An input file is needed for filtering")
        if output_file is None:
            output_file = "{}-filtered.arff".format(str(input_file_name.rstrip(".arff")))
        if self.verbose:
            print("Filtering input data...")
        options = ["java", "-Xmx{}M".format(str(self.max_memory))]
        if self.classpath is not None:
            options.extend(["-cp", self.classpath])
        options.extend(filter_options)
        options.extend(["-i", input_file_name, "-o", output_file, "-c", class_column])
        process_output, run_time = run_process(options)
        if self.verbose:
            print("Filtering complete (time taken = {:.2f}s)".format(run_time))
        return output_file

    def split(self, input_file_name=None, training_percentage=67, randomise=True, seed=None):
        if input_file_name is None:
            raise WekapyException("An input file is needed for filtering")
        if not isinstance(training_percentage, int):
            raise WekapyException("'training_percentage' argument must be of type (int).")
        options = ["java", "-Xmx{}M".format(str(self.max_memory))]
        if self.classpath is not None:
            options.extend(["-cp", self.classpath])
        if randomise is True and seed is None:
            seed = random.randint(0, 1000)
        if randomise is True:
            if self.verbose:
                print("Randomising data order...")
            output_file = "{}-randomised.arff".format(str(input_file_name.rstrip(".arff")))
            options.extend(
                ["weka.filters.unsupervised.instance.Randomize", "-S", str(seed), "-i", input_file_name, "-o",
                 output_file])
            process_output, run_time = run_process(options)
            input_file_name = output_file
            if self.verbose:
                print("Randomisation complete (time taken = {:.2f}s).".format(run_time))
        if self.verbose:
            print("Beginning split...\nCreating training set...")
        output_file = "{}-training.arff".format(str(input_file_name.rstrip(".arff")))
        options.extend(
            ["weka.filters.unsupervised.instance.RemovePercentage", "-P", str(training_percentage), "-V", "-i",
             input_file_name, "-o", output_file])
        process_output, run_time_training = run_process(options)
        if self.verbose:
            print("Creating testing set...")
        output_file = "{}-testing.arff".format(str(input_file_name.rstrip(".arff")))
        options.extend(["weka.filters.unsupervised.instance.RemovePercentage", "-P", str(training_percentage), "-i",
                        input_file_name, "-o", output_file])
        process_output, run_time_testing = run_process(options)
        if self.verbose:
            print("Split complete (time taken = {:.2f}s).".format(run_time_training + run_time_testing))


# Model class
#
# Used externally, and is the main class for use with this library.
# The Model class should be instantiated as the first stage, from which it can be trained
# and/or tested.
# Instantiate with a classifier_type (and any optional arguments)
class Model:
    def __init__(self, classifier_type=None, max_memory=1500, classpath=None, verbose=False):
        if classifier_type is None or not isinstance(classifier_type, str):
            raise WekapyException("A classifier type is required for construction.")
        if not isinstance(max_memory, int):
            raise WekapyException("'max_memory' argument must be of type (int).")
        self.id = uuid.uuid4()
        self.model_dir = "wekapy_data/models"
        self.arff_dir = "wekapy_data/arff"
        self.classpath = classpath
        self.classifier = classifier_type
        self.max_memory = max_memory
        self.training_instances = []
        self.testing_instances = []
        self.predictions = []
        self.time_taken = 0.0
        self.verbose = verbose
        self.trained = False
        self.model_file = None
        self.training_file = None
        self.test_file = None
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.arff_dir):
            os.makedirs(self.arff_dir)

    # Generate an ARFF file from a list of instances
    def create_arff(self, instances, data_type):
        output_arff = open(os.path.join(self.arff_dir, "{}-{}.arff".format(str(self.id), data_type)), "w")
        output_arff.write("@relation " + str(self.id) + "\n")
        for i, instance in enumerate(instances):
            if i == 0:
                for feature in instance.features:
                    output_arff.write("\t@attribute " + feature.name + " " + str(feature.possible_values) + "\n")
                output_arff.write("\n@data\n")
            str_to_write = ""
            for j, feature in enumerate(instance.features):
                if j == 0:
                    str_to_write = str_to_write + str(feature.value)
                else:
                    str_to_write = str_to_write + "," + str(feature.value)
            output_arff.write(str_to_write + "\n")
        output_arff.close()
        if data_type == "training":
            self.training_file = self.arff_dir + "/" + str(self.id) + "-" + data_type + ".arff"
        if data_type == "test":
            self.test_file = self.arff_dir + "/" + str(self.id) + "-" + data_type + ".arff"

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
    def train(self, training_file=None, instances=None, save_as=None, folds=10):
        if self.verbose:
            print("Training your classifier...")
        if save_as is None:
            save_as = self.model_dir + "/" + str(self.id) + ".model"
        if len(self.training_instances) == 0:  # if add_train_instance not called:
            if training_file is None and instances is None:
                raise WekapyException(
                    "Please provide some train instances either by naming an ARFF train_set, providing a list of Instances, or calling add_train_instance().")
            if training_file is None:
                self.create_arff(instances, "training")
            if instances is None:
                self.training_file = training_file
        if len(self.training_instances) > 0:  # if add_train_instance called:
            if training_file is None and instances is None:
                self.create_arff(self.training_instances, "training")
            # Prioritise adding features passed at call time
            if training_file is None and instances is not None:
                self.create_arff(instances, "training")
            # Prioritise ARFF file passed at calltime
            if instances is None and training_file is not None:
                self.training_file = training_file

        self.model_file = save_as
        options = ["java", "-Xmx{}M".format(str(self.max_memory))]
        if self.classpath is not None:
            options.extend(["-cp", self.classpath])
        options.extend(
            ["weka.classifiers." + self.classifier, "-x", str(folds), "-t", self.training_file, "-d", save_as])
        process_output, self.time_taken = run_process(options)
        self.trained = True
        if self.verbose:
            print("Training complete (time taken = {:.2f}s).".format(self.time_taken))

    # Generate predictions from the trained model from test features in an ARFF file
    def test(self, test_file=None, instances=None, model_file=None):
        if self.verbose:
            print("Generating predictions for your test set...")
        if model_file is not None:
            self.load_model(model_file)
        if not self.trained:
            raise WekapyException("The classifier has not yet been trained. Please call train() first")
        if len(self.testing_instances) == 0:
            if test_file is None and instances is None:
                raise WekapyException(
                    "Please provide some test instances either by naming an ARFF test_set, providing a list of Instances, or calling add_test_instance().")
            if test_file is None:
                self.create_arff(instances, "test")
            if instances is None:
                self.test_file = test_file
        if len(self.testing_instances) > 0:
            if test_file is None and instances is None:
                self.create_arff(self.testing_instances, "test")
            if test_file is None and instances is not None:
                self.create_arff(instances, "test")
            if instances is None and test_file is not None:
                self.test_file = test_file

        options = ["java", "-Xmx{}M".format(str(self.max_memory))]
        if self.classpath is not None:
            options.extend(["-cp", self.classpath])
        options.extend(["weka.classifiers." + self.classifier, "-T", self.test_file, "-l", self.model_file, "-p", "0"])
        process_output, self.time_taken = run_process(options)

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
                if "+" in pred[3]:
                    error = True
                    prob = float(pred[4])
                else:
                    prob = float(pred[3])
                prediction = Prediction(index, ob_cat, ob_val, p_cat, p_val, error, prob)
                instance_predictions.append(prediction)
        self.predictions = instance_predictions
        if self.verbose:
            print("Testing complete (time taken = {:.2f}s).".format(self.time_taken))
        return instance_predictions


class WekapyException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message
