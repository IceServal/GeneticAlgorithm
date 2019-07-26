# Genetic Algorithm

This is the Python implementation of genetic algorithm. It can be used without installing package and the configuration are written in file with `json` format thus easy to reuse or modify.

## Configuration Format

* Group Configuration
  * **initial individual number** (type: int, range: [1, +∞)):
  
    The initial quantity of the individuals in the group.
  * **group birth rate** (type: float, range: (0, +∞)):
  
    The percentage of newborns of the group in each generation.
  * **keep best individuals rate** (type: float, range: [0, +∞)):
  
    The percentage of best individuals of the group who will survive to the next generation.
  * **mate candidates number** (type: int, range: [1, +∞]):

    When we get newborns, we need to select a couple of individuals to give it birth. This parameter determines that how many candidates will be selected to compete for procreation right.
* Genome Configuration
  * **names**:
  * **types**:
  * **specs**:
  * **crossover method** (type: str, range: ["exchange", "integrate"]):
  
    The crossover method used in performing crossover on gene pair.
  * **crossover rate** (type: float, range: [0, 1]):
  
    The crossover rate of all the genes.
  * **mutate rate** (type: float, range: [0, 1]):
  
    The mutate rate of all the genes.
  * **mutate rate decay rate** (type: float, range: [0, 1]):
  
    The mutate rate will decay after each time of group evolving. The mutate rate will updated with old mutate rate multiplied with decay rate.
  * **mutate scale** (type: float, range:[0, 1]):

    The maximum scale scope will be limited in range `[-mutate_scale*gene_max_value, mutate_scale*gene_max_value]`
  * **mutate scale decay rate** (type: float, range: [0, 1]):

    After each time of evolving, the mutate scale will be decreased by multiplying the decay rate to make sure the convergency of evolving progress.

## Usage

```python
from genetic_algorithm import Group, Genome, Administrator

# Create the Population Group
group = Group(group_config_path)
# Create the Genome Prototype
genome = Genome(genome_config_path)
# Create the Evolve Administrator
admin = Administrator(group, genome)

# The Example of Defining a fitness function
def fitness_function(individuals, environment):
    """

    :param individuals: the individuals created or modified by
      Genome class.
    :param environment: user defined environment which will be
      used for evaluating the scores of individuals.
    :rtype: array-like score list.

    """
    scores = []
    for indiv in individuals:
        model = model_creator(indiv)
        scores.append(model.eval(environment))

    return scores

# Initial Evolve Administrator
admin.fitness_eval = fitness_function
admin.environment = eval_environment

# Initial All the Variables and Evaluate the Origin Individuals
admin.global_variables_initialize()

# Evolve Group and Get the Information of the Best Individual
admin.evolve()
admin.get_winner_board()
```

## Details Illustration
