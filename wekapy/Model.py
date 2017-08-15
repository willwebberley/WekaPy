# Model class
#
# Used externally, and is the main class for use with this library.
# The Model class should be instantiated as the first stage, from which it can be trained
# and/or tested.
# Instantiate with a classifier_type (and any optional arguments)

from wekapy.Prediction import Prediction
from wekapy.Instance import Instance
from wekapy.Helpers import run_process
from wekapy.WekaPyException import WekaPyException
import os
import uuid


class Model:
    def __init__(self, classifier_type=None, max_memory=1500, classpath=None, verbose=False):
        if classifier_type is None or not isinstance(classifier_type, str):
            raise WekaPyException("A classifier type is required for construction.")
        if not isinstance(max_memory, int):
            raise WekaPyException("'max_memory' argument must be of type (int).")
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
            raise WekaPyException("Your model could not be found.")

    # Add a training instance to the model.
    def add_train_instance(self, instance):
        if isinstance(instance, Instance):
            self.training_instances.append(instance)
        else:
            raise WekaPyException("Argument 'instance' must be of type Instance.")

    # Add a testing instance to the model.
    def add_test_instance(self, instance):
        if isinstance(instance, Instance):
            self.testing_instances.append(instance)
        else:
            raise WekaPyException("Argument 'instance' must be of type Instance.")

    # Train the model with the chosen classifier from features in an ARFF file
    def train(self, training_file=None, instances=None, save_as=None, folds=10):
        if self.verbose:
            print("Training your classifier...")
        if save_as is None:
            save_as = self.model_dir + "/" + str(self.id) + ".model"
        if len(self.training_instances) == 0:  # if add_train_instance not called:
            if training_file is None and instances is None:
                raise WekaPyException(
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
            raise WekaPyException("The classifier has not yet been trained. Please call train() first")
        if len(self.testing_instances) == 0:
            if test_file is None and instances is None:
                raise WekaPyException(
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
