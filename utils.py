"""The implementation of the genetic algorithm.

@author: icemaster
@create: 2019-7-17
@update: 2019-11-6

"""

import numpy as np


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

    return tower_binary_search(prob_tower, lottery)


def tower_binary_search(sorted_array, target):
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
