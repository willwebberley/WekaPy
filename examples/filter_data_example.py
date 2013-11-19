# This example demonstrates:
#   - creating a Filter object
#   - filtering an input data set
#   - splitting an input data set into training and testing sets

from wekapy import *

# create new filter instance
data_filter = Filter()

# filter the input data
filtered_file = data_filter.filter(filter_type=["weka.filters.supervised.attribute.AttributeSelection", "-E", "weka.attributeSelection.CfsSubsetEval ", "-S", "weka.attributeSelection.BestFirst", "-D", "1", "-N", "5"], input_file_name="filter_test.arff")

# create a testing/training split
data_filter.split(input_file_name=filtered_file)
