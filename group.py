"""The definition of the population. A population group contains several
individuals which belong to same specie. Also some properties, such as
group size and birth rate, are attached to the group.

@author: icemaster
@create: 2019-7-17
@update: 2019-11-12

"""

import json


# pylint: disable=too-few-public-methods
class Group:
    """The definition of the group."""
    def __init__(self, param_path):
        with open(param_path, "r") as param_file:
            param = json.load(param_file)
        self.init_num = param["initial individual number"]
        self.birth_rate = param["group birth rate"]
        self.keep_rate = param["keep best individuals rate"]
        self.matepool_size = param["mate candidates number"]

        self.members = None
        self.size = 0
        self.growth_rate = self.birth_rate + self.keep_rate
