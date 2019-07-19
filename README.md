# Genetic Algorithm

This is the Python implementation of genetic algorithm. It can be used without installing package and the configuration are written in file with `json` format thus easy to reuse or modify.

## Configuration Format

* Group Configuration
  * **initial individual number**: The initial quantity of the individuals in the group. (type: int, range: [1, +∞))
  * **group birth rate**: (type: float, range: (0, +∞))
  * **keep best individuals rate**: (type: float, range: [0, +∞))
  * **mate candidates number**: (type: int, range: [1, +∞])
* Genome Configuration
  * **names**:
  * **types**:
  * **specs**:
  * **crossover method**: The crossover method used in performing crossover on gene pair. (type: str, range: ["exchange", "integrate"])
  * **crossover rate**: The crossover rate of all the genes. (type: float, range: [0, 1])
  * **mutate rate**: The mutate rate of all the genes. (type: float, range: [0, 1])
  * **mutate rate decay rate**: The mutate rate will decay after each time of group evolving. The mutate rate will updated with old mutate rate multiplied with decay rate. (type: float, range: [0, 1])
  * **mutate scale**: (type: , range:)
  * **mutate scale decay rate**: (type: , range:)

## Usage

```python
from genetic_algorithm import Group, Genome, Administrator

# Create the population group
group = Group(group_config_path)
# Create the genome prototype
genome = Genome(genome_config_path)
# Create the evolve administrator
admin = Administrator(group, genome)

# Define fitness function
def fitness_function(individuals, environment):
    scores = []
    for indiv in individuals:
        model = model_creator(indiv)
        scores.append(model.eval(environment))

    return scores

# Initial evolve administrator
admin.fitness_eval = fitness_function
admin.environment = eval_environment
admin.global_variables_initialize()

# Evolve group and get the information of the best individual
admin.evolve()
winner_board = admin.get_winner_board()
```

## Details Illustration
