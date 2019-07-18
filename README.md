# Genetic Algorithm

This is the Python implementation of genetic algorithm. It can be used without installing package and the configuration are written in file with `json` format thus easy to reuse or modify.

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
