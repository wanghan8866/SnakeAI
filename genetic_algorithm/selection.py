# genetic_algorithm/Selection.py
import numpy as np
from typing import List
from .population import Population
from .individual import Individual


def ellitism_selection(population: Population, num_individuals: int) -> List[Individual]:
    individuals = sorted(population.individuals, key = lambda individual: individual.fitness, reverse=True)
    return individuals[:num_individuals]

def roulette_wheel_selection(population: Population, num_individuals: int) -> List[Individual]:
    selection = []
    wheel = np.sum(individual.fitness for individual in population.individuals)
    for _ in range(num_individuals):
        pick = np.random.uniform(0, wheel)
        current = 0
        for individual in population.individuals:
            current += individual.fitness
            if current > pick:
                selection.append(individual)
                break

    return selection

def tournament_selection(population: Population, num_individuals, tournament_size: int) -> List[Individual]:
    selection = []
    for _ in range(num_individuals):
        tournament = np.random.choice(population.individuals, tournament_size)
        best_from_tournament = max(tournament, key = lambda individual: individual.fitness)
        selection.append(best_from_tournament)

    return selection