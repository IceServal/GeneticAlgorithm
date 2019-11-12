"""The definition of the genome. A genome should not only contains the
integral genetic information, but also record the interaction method
between the genomes from parents to generate the offspring. We also put
the function which can create several random individuals according to
the genetic information in this class.

@author: icemaster
@create: 2019-7-17
@update: 2019-11-12

"""

import json
import functools
import numpy as np

# pylint: disable=wrong-import-order
from copy import deepcopy
from .utils import roulette_wheel_selection


# pylint: disable=too-many-instance-attributes
class Genome:
    """The definition of the genome."""
    def __init__(self, pattern_path, gene_editor=None):
        with open(pattern_path, "r") as pattern_file:
            pattern = json.load(pattern_file)
        self.names = pattern["names"]
        self.types = pattern["types"]
        self.specs = pattern["specs"]
        self.editor = gene_editor
        if self.editor is None:
            self.editor = {"int": _randint, "float": _randfloat,
                           "bool": _randbool, "sand": _randsand}

        self.cross_method = pattern["crossover method"]
        self.cross_rate = pattern["crossover rate"]
        self.mutate_rate = pattern["mutate rate"]
        self.mutate_rate_decay = pattern["mutate rate decay rate"]
        self.mutate_scale = pattern["mutate scale"]
        self.mutate_scale_decay = pattern["mutate scale decay rate"]

        self.size = len(self.names)
        self.cross_funcs = {"exchange": self._exchange_crossover,
                            "integrate": self._integrate_crossover}

    def create_individuals(self, group_size):
        """Create several individuals with the self genome pattern, the
        number of individuals are in accordance with the group size.

        :param group_size: the number of individuals will be generated.

        """
        group_members = []
        for _ in range(group_size):
            individual = dict()
            for name, type_, spec in zip(self.names, self.types, self.specs):
                individual[name] = self.editor[type_](spec)
            group_members.append(individual)

        return np.array(group_members)

    def get_pair_num(self, newbirth_n):
        """Return the necessary pair number of the pairs which will
        crossover with each other and generate descendants.

        :param newbirth_n: the new birth individuals number.

        """
        if self.cross_method == "exchange":
            return int(np.ceil(newbirth_n/2))
        if self.cross_method == "integrate":
            return int(np.ceil(newbirth_n/2))
        return newbirth_n

    def crossover(self, individual_pair):
        """Crossover the genomes of input gonome pair according to the
        select method and the pattern saved in current class instance.

        :param individual_pair: a list or tuple contains a pair of
          individuals.

        """

        return self.cross_funcs[self.cross_method](individual_pair)

    def mutate(self, individual):
        """Mutate the genome according to mutate rate and mutate step
        size.

        :param individual: the individual which will be mutated.

        """
        individual = deepcopy(individual)
        bool_spec = {"shape": None, "tprob": self.cross_rate}
        mutator = {"float": _float_mutate,
                   "int": _int_mutate,
                   "bool": _bool_mutate,
                   "sand": _sand_mutate}
        for name, type_, spec in zip(self.names, self.types, self.specs):
            do_mutation = _randbool(bool_spec)
            if do_mutation:
                gene = individual[name]
                individual[name] = mutator[type_](gene, spec,
                                                  self.mutate_scale)

        return individual

    def mutate_decay(self):
        """Do one time mutation rate and scale decay."""
        self.mutate_rate *= self.mutate_rate_decay
        self.mutate_scale *= self.mutate_scale_decay

    def _exchange_crossover(self, individual_pair):
        """According to crossover probability to exchange the genes
        between the individuals of a single pair. Each time, two
        descendants will be generated.

        :param individual_pair: a list contains two individuals which
          will crossover with each other.

        """
        descendants = deepcopy(individual_pair)
        bool_spec = {"shape": None, "tprob": self.cross_rate}
        for name in self.names:
            do_crossover = _randbool(bool_spec)
            if do_crossover:
                gene1 = descendants[0][name]
                gene2 = descendants[1][name]
                descendants[0][name] = gene2
                descendants[1][name] = gene1

        return descendants

    def _integrate_crossover(self, individual_pair):
        """According to corssover probability to integrate several genes
        and make two individuals get same resulted gene. The integrate
        rule can be relative complicated. Here, I just compute the mean
        value if target is float and random select a value in the
        interval if target is int or bool.

        :param individual_pair: a list contains two individuals which
          will crossover with each other.

        """
        descendants = deepcopy(individual_pair)
        bool_spec = {"shape": None, "tprob": self.cross_rate}
        integrator = {"float": lambda x, y: (x[0] + x[1])/2,
                      "int": _int_gene_integrate,
                      "bool": _bool_gene_integrate,
                      "sand": _sand_gene_integrate}
        for name, type_, spec in zip(self.names, self.types, self.specs):
            do_crossover = _randbool(bool_spec)
            if do_crossover:
                gene1 = descendants[0][name]
                gene2 = descendants[1][name]
                fusion = integrator[type_]([gene1, gene2], spec)
                descendants[0][name] = fusion
                descendants[1][name] = fusion

        return descendants


def _randint(spec):
    """Generate a random integer according to the specification.

    :param sepc: a dict contains the specification including the range
      and shape.

    """

    return np.random.randint(spec["range"][0], spec["range"][1] + 1,
                             size=spec["shape"])


def _randfloat(spec):
    """Generate a random float according to the specification.

    :param sepc: a dict contains the specification including the range
      and shape.

    """
    offset = spec["range"][0]
    scale = spec["range"][1] - spec["range"][0]

    return np.random.random_sample(spec["shape"])*scale + offset


def _randbool(spec):
    """Generate a random bool according to the specificaton.

    :param spec: a dict contains the specification including the shape
      and the true occurance probability.

    """

    return np.random.random_sample(spec["shape"]) < spec["tprob"]


def _randsand(spec):
    """Random select a value from a discrete value set given by the
    sepcification. If the weights of different values are given, the
    random process will be influenced.

    :param spec: a dict contains the specification including the
      discrete values and the corresponding weights.

    """
    values = spec["values"]
    weights = [1 for _ in range(len(values))]
    if spec["weights"] is not None:
        weights = spec["weights"]

    if spec["shape"] is not None:
        flatten_length = functools.reduce(lambda x, y: x*y, spec["shape"])
        sandpile = []
        for _ in range(flatten_length):
            sandpile.append(values[roulette_wheel_selection(weights)])

        return np.reshape(sandpile, spec["shape"])

    return values[roulette_wheel_selection(weights)]


def _float_mutate(gene, spec, mutate_scale):
    """Mutate a float type gene.

    :param gene: the gene will be mutated.
    :param spec: the specification of relative gene.
    :param mutate_scale: the maximum mutation ratio with repect to the
      full range.

    """
    min_ = spec["range"][0]
    max_ = spec["range"][1]
    amplitude = (max_ - min_)*mutate_scale
    perturb_spec = {"range": [-amplitude, amplitude], "shape": spec["shape"]}
    perturbation = _randfloat(perturb_spec)
    gene += perturbation
    if spec["shape"] is None:
        gene = max(min(gene, max_), min_)
    else:
        gene[gene < min_] = min_
        gene[gene > max_] = max_

    return gene


def _int_mutate(gene, spec, mutate_scale):
    """Mutate a int type gene.

    :param gene: the gene will be mutated.
    :param spec: the specification of relative gene.
    :param mutate_scale: the maximum mutation ratio with repect to the
      full range.

    """
    min_ = spec["range"][0]
    max_ = spec["range"][1]
    amplitude = int((max_ - min_ + 1)*mutate_scale)
    perturb_spec = {"range": [-amplitude, amplitude], "shape": spec["shape"]}
    perturbation = _randint(perturb_spec)
    gene += perturbation
    if spec["shape"] is None:
        gene = max(min(gene, max_), min_)
    else:
        gene[gene < min_] = min_
        gene[gene > max_] = max_

    return gene


# pylint: disable=assignment-from-no-return
def _bool_mutate(gene, spec, mutate_scale):
    """Mutate a bool type gene.

    :param gene: the gene will be mutated.
    :param spec: the specification of relative gene.
    :param mutate_scale: the maximum mutation ratio with repect to the
      full range.

    """
    perturb_spec = {"shape": spec["shape"], "tprob": mutate_scale}
    perturbation = _randbool(perturb_spec)
    if spec["shape"] is None:
        gene = bool(gene - perturbation)
    else:
        gene = np.logical_xor(gene, perturbation)

    return gene


def _sand_mutate(gene, spec, mutate_scale):
    """Mutate a sand type gene.

    :param gene: the gene will be mutated.
    :param spec: the specification of relative gene.
    :param mutate_scale: the probability that mutation does not lead to
      a changing on gene sand value.

    """
    weights = [1 for _ in range(len(spec["values"]))]
    if spec["weights"] is not None:
        weights = spec["weights"]
    value_weight_map = dict(zip(spec["values"], weights))

    perturb_spec = {"shape": spec["shape"], "tprob": mutate_scale}
    perturbation = _randbool(perturb_spec)
    if spec["shape"] is None:
        if perturbation:
            local_value_weight_map = deepcopy(value_weight_map)
            local_value_weight_map.pop(gene)
            spec_x = {"values": list(local_value_weight_map.keys()),
                      "weights": list(local_value_weight_map.values()),
                      "shape": None}
            gene = _randsand(spec_x)
    else:
        gene_flatten = np.reshape(gene, [-1])
        perturb_flatten = np.reshape(perturbation, [-1])
        gene = []
        for sand, perturb in zip(gene_flatten, perturb_flatten):
            if perturb:
                local_value_weight_map = deepcopy(value_weight_map)
                local_value_weight_map.pop(sand)
                spec_x = {"values": list(local_value_weight_map.keys()),
                          "weights": list(local_value_weight_map.values()),
                          "shape": None}
                gene.append(_randsand(spec_x))
            else:
                gene.append(sand)

        gene = np.reshape(gene, spec["shape"])

    return gene


def _int_gene_integrate(gene_pair, spec):
    """Integrate two int type gene.

    :param gene_pair: a pair of int type genes.
    :param spec: the specification of the input gene.

    """
    fusion = None
    gene1, gene2 = gene_pair
    if spec["shape"] is None:
        min_ = min(gene1, gene2)
        max_ = max(gene1, gene2)
        fusion = np.random.randint(min_, max_ + 1)
    else:
        fusion = []
        flatten1 = np.reshape(gene1, [-1])
        flatten2 = np.reshape(gene2, [-1])
        gene_size = flatten1.shape[0]
        for i in range(gene_size):
            min_ = min(flatten1[i], flatten2[i])
            max_ = max(flatten1[i], flatten2[i])
            fusion.append(np.random.randint(min_, max_ + 1))

        fusion = np.reshape(fusion, spec["shape"])

    return fusion


def _bool_gene_integrate(gene_pair, spec):
    """Integrate two bool type gene.

    :param gene_pair: a pair of bool type genes.
    :param spec: the specification of the input gene.

    """
    fusion = None
    gene1, gene2 = gene_pair
    if spec["shape"] is None:
        fusion = _randbool({"shape": None, "tprob": (gene1 + gene2)/2})
    else:
        fusion = []
        flatten1 = np.reshape(gene1, [-1])
        flatten2 = np.reshape(gene2, [-1])
        gene_size = flatten1.shape[0]
        for i in range(gene_size):
            bool1 = bool(flatten1[i])
            bool2 = bool(flatten2[i])
            bool_kid = _randbool({"shape": None, "tprob": (bool1 + bool2)/2})
            fusion.append(bool_kid)

        fusion = np.reshape(fusion, spec["shape"])

    return fusion


def _sand_gene_integrate(gene_pair, spec):
    """Integrate two sand type gene.

    :param gene_pair: a pair of bool type genes.
    :param sepc: the specification of the input gene.

    """
    fusion = None
    gene1, gene2 = gene_pair
    weights = [1 for _ in range(len(spec["values"]))]
    if spec["weights"] is not None:
        weights = spec["weights"]
    value_weight_map = dict(zip(spec["values"], weights))

    if spec["shape"] is None:
        fusion = _randsand({"values": [gene1, gene2],
                            "weights": [value_weight_map[gene1],
                                        value_weight_map[gene2]],
                            "shape": None})
    else:
        fusion = []
        flatten1 = np.reshape(gene1, [-1])
        flatten2 = np.reshape(gene2, [-1])
        gene_size = flatten1.shape[0]
        for i in range(gene_size):
            sand1 = flatten1[i]
            sand2 = flatten2[i]
            sand_kid = _randsand({"values": [sand1, sand2],
                                  "weights": [value_weight_map[sand1],
                                              value_weight_map[sand2]],
                                  "shape": None})
            fusion.append(sand_kid)

        fusion = np.reshape(fusion, spec["shape"])

    return fusion
