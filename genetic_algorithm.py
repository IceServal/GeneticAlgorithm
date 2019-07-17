"""The implementation of the genetic algorithm"""

import json
import functools
from copy import deepcopy

import numpy as np
from numpy import concatenate as npconcat


# pylint: disable=too-few-public-methods
class Group:
    """The parameters used in genetic algorithm."""
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


# pylint: disable=too-many-instance-attributes
class Genome:
    """The pattern of gene including data type and range."""
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


class Administrator:
    """The genetic algorithm class which can be used as a template
    working for standard genetic envolution search problems."""
    def __init__(self, group, genome, environment=None, fitness_function=None):
        self.group = group
        self.genome = genome
        self.environment = environment
        self.fitness_eval = fitness_function
        self.fitness_scores = None

    def global_variables_initialize(self):
        """Generate several individuals according to group parameters
        and genome pattern. The fitness scores of the newly generated
        individuals are also computed."""
        self.group.size = self.group.init_num
        self.group.members = self.genome.create_individuals(self.group.size)
        self.fitness_scores = np.array(self.fitness_eval(self.group.members,
                                                         self.environment))

    def evolve(self):
        """Envove one time and update the group."""
        newbirth_n = int(np.round(self.group.size*self.group.birth_rate))
        keepalive_n = int(np.round(self.group.size*self.group.keep_rate))
        pair_num = self.genome.get_pair_num(newbirth_n)
        cross_pairs = _get_cross_pairs(self.fitness_scores, pair_num,
                                       self.group.matepool_size)
        descendants = self._mutate(self._crossover(cross_pairs))
        keepmembers, keepscores = self._best_individuals_select(keepalive_n)

        descendant_scores = np.array(self.fitness_eval(descendants,
                                                       self.environment))
        self.fitness_scores = npconcat((descendant_scores, keepscores))

        self.group.members = npconcat((descendants, keepmembers))
        self.group.size = newbirth_n + keepalive_n
        self.genome.mutate_decay()

    def get_winner_board(self):
        """Return the winner board, which is the encapsuled winner
        information including the winner private individual information
        and its score performance."""
        winner_info = self._best_individuals_select(1)
        winner_board = {"indiv": winner_info[0][0], "score": winner_info[1][0]}

        return winner_board

    def _crossover(self, cross_pairs):
        """Do crossover in each selected pairs to generator a batch of
        descendant individuals.

        :param corss_pairs: a list contains several lists, each list of
          those contains two indexes indexing the select individuals.

        """
        descendants = []
        for pair in cross_pairs:
            descendants.extend(self.genome.crossover(self.group.members[pair]))

        return np.array(descendants)

    def _mutate(self, individuals):
        """Do mutation on individuals.

        :param individuals: the individuals which will get mutated.

        """
        descendants = []
        for indiv in individuals:
            descendants.append(self.genome.mutate(indiv))

        return np.array(descendants)

    def _best_individuals_select(self, window_size):
        """Select best individuals and these individuals will be
        reserved to next generation of the group without crossover and
        mutation.

        :param window_size: how many best individuals will be selected.

        """
        if window_size == 1:
            winner_idx = np.argmax(self.fitness_scores)
            winner = np.array([self.group.members[winner_idx]])
            winner_score = np.array([self.fitness_scores[winner_idx]])

            return winner, winner_score

        indexed_scores = list(zip(range(self.group.size), self.fitness_scores))
        sorted_scores = sorted(indexed_scores, key=lambda x: x[1],
                               reverse=True)
        select_indexes = np.int32(np.array(sorted_scores)[:window_size, 0])

        winners = self.group.members[select_indexes]
        winner_scores = self.fitness_scores[select_indexes]

        return winners, winner_scores


def roulette_wheel_selection(probabilities):
    """Random select several candidates according to fitness scores and
    return the best one of the candidates.

    :param fitness_scores: the fitness scores given by fitness feedback
      function.
    :param select_width: to generate an individual, how many individuals
      should be random selected from group and compete to win a chance.

    """
    prob_sum = 0
    prob_tower = []
    for prob in probabilities:
        prob_sum += prob
        prob_tower.append(prob_sum)

    lottery = np.random.rand()*prob_sum

    return _tower_binary_search(prob_tower, lottery)


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


def _get_cross_pairs(fitness_scores, pair_num, select_width):
    """Use roulette wheel selection algorithm to select several pairs.
    The genes of theses pairs of individuals will crossover with each
    other and the gene of their descendants will mutate.

    :param fitness_scores: the fitness scores given by fitness feedback
      function.
    :param pair_num: how many pairs need to be generated.
    :param select_width: to generate an individual, how many individuals
      should be random selected from group and compete to win a chance.
    :return: a list contain lists. Each list are composed of two
      individuals' indexes.

    """
    cross_pairs = []
    for _ in range(pair_num):
        cross_pairs.append([_get_mate(fitness_scores, select_width),
                            _get_mate(fitness_scores, select_width)])

    return cross_pairs


def _get_mate(fitness_scores, select_width):
    """Use roulette wheel selection algorithm to select several
    individuals as candidates and select best on of them as the winner.
    The gene of winner will be crossed with another individual, namely,
    it will be a mate.

    :param fitness_scores: the fitness scores given by fitness feedback
      function.
    :param select_width: to generate an individual, how many individuals
      should be random selected from group and compete to win a chance.

    """
    candidates = []
    election_scores = []
    for _ in range(select_width):
        luckydog = roulette_wheel_selection(fitness_scores)
        candidates.append(luckydog)
        election_scores.append(fitness_scores[luckydog])
    winner = candidates[np.argmax(election_scores)]

    return winner


def _tower_binary_search(sorted_array, target):
    """A especially binary search algorithm implementation for roulette
    wheel selection.

    :param sorted_array: a sorted array used for search the target.
    :param target: you know, the target value. In most of the cases, it
      has same type with the elements in the sorted array.

    """
    if target >= sorted_array[-1]:
        print("Invalid target value.")
        return len(sorted_array) - 1
    low, high = 0, len(sorted_array)
    while low <= high:
        mid = int((low + high)/2)
        mid_value = sorted_array[mid]
        if mid == 0:
            if target < mid_value:
                return mid
            low = mid + 1
        else:
            mid_left_value = sorted_array[mid - 1]
            if mid_left_value <= target < mid_value:
                return mid
            if target < mid_left_value:
                high = mid - 1
            else:
                low = mid + 1
