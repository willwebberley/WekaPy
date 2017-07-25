# Filter class
#
# Used to filter/pre-process data using one of the weka.filters classes.

from wekapy.Helpers import run_process
from wekapy.WekaPyException import WekaPyException
import uuid
import random


class Filter:
    def __init__(self, max_memory=1500, classpath=None, verbose=False):
        if not isinstance(max_memory, int):
            raise WekaPyException("'max_memory' argument must be of type (int).")
        self.classpath = classpath
        self.max_memory = max_memory
        self.id = uuid.uuid4()
        self.verbose = verbose

    def filter(self, filter_options=None, input_file_name=None, output_file=None, class_column="last"):
        if filter_options is None:
            raise WekaPyException("A filter type is required")
        if input_file_name is None:
            raise WekaPyException("An input file is needed for filtering")
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
            raise WekaPyException("An input file is needed for filtering")
        if not isinstance(training_percentage, int):
            raise WekaPyException("'training_percentage' argument must be of type (int).")
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
