"""The implementation of the genetic algorithm.

@author: icemaster
@create: 2019-7-17
@update: 2019-11-6

"""

import numpy as np

from numpy import concatenate as npconcat
from .utils import roulette_wheel_selection


class Controller:
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
