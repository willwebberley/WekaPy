# Instance class
#
# Used internally and externally to represent a set of Feature objects.
# Essentially, an Instance object just maintains a list of Features.

from wekapy.WekaPyException import WekaPyException
from wekapy.Feature import Feature


class Instance:
    def __init__(self, features=None):
        self.features = features
        if features is None:
            self.features = []

    def add_feature(self, feature):
        if isinstance(feature, Feature):
            self.features.append(feature)
        else:
            raise WekaPyException("Argument 'feature' must be of type Feature.")

    def add_features(self, features_list):
        for feature in features_list:
            if isinstance(feature, Feature):
                self.features.append(feature)
            else:
                raise WekaPyException("Argument 'feature' must be of type Feature.")
