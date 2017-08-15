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
