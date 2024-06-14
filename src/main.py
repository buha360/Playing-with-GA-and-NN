import csv

import numpy as np
import pandas as pd
import time

# Itt kezdődik az időmérés
global_start_time = time.time()
bounds = (-10, 10)


# Rastrigin Function
def rastrigin(X):
    A = 10
    n = len(X)
    return A * n + sum([(x**2 - A * np.cos(2 * np.pi * x)) for x in X])


# Modified Rastrigin Function for two variables
def rastrigin_modified(x, y):
    return rastrigin([x, y])


# Rastrigin Function n dimension
def n_dim_rastrigin(X):
    A = 10
    n = len(X)
    return A * n + sum([(x**2 - A * np.cos(2 * np.pi * x)) for x in X])


# Itt történik az eldőntése, hogy n dimenzios rastrigin vagy sem
def fitness_function(individual, function):
    if function == n_dim_rastrigin:
        return function(individual)  # Az egész tömböt átadjuk, ha az n dimenziós Rastrigin függvényt használjuk
    else:
        return function(*individual)  # Különben kicsomagoljuk az elemeket


# Booth Function
def booth(x, y):
    return (x + 2 * y - 7)**2 + (2 * x + y - 5)**2


# Levi Function
def levi(x, y):
    return np.sin(3 * np.pi * x)**2 + (x - 1)**2 * (1 + np.sin(3 * np.pi * y)**2) + \
           (y - 1)**2 * (1 + np.sin(2 * np.pi * y)**2)


# Initialize Population
def initialize_population(population_size, dimension, bounds):
    lower_bound, upper_bound = bounds
    return np.random.uniform(lower_bound, upper_bound, (population_size, dimension))


# Rank Selection
def rank_selection(population, fitness_values, num_selected):
    ranked_indices = np.argsort(fitness_values)
    selected_indices = ranked_indices[:num_selected]
    return population[selected_indices]


# Crossover
def crossover(parent1, parent2):
    midpoint = len(parent1) // 2
    offspring1 = np.concatenate([parent1[:midpoint], parent2[midpoint:]])
    offspring2 = np.concatenate([parent2[:midpoint], parent1[midpoint:]])
    return offspring1, offspring2


# Create Offspring
def create_offspring(selected_population, num_offspring):
    offspring = []
    num_parents = len(selected_population)
    while len(offspring) < num_offspring:
        parent_indices = np.random.choice(num_parents, 2, replace=False)
        parents = [selected_population[idx] for idx in parent_indices]
        off1, off2 = crossover(parents[0], parents[1])
        offspring.extend([off1, off2])
    return np.array(offspring[:num_offspring])


# Mutáció
def mutate(individual, mutation_rate, step_size):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] += np.random.uniform(-step_size, step_size)
            individual[i] = np.clip(individual[i], bounds[0], bounds[1])
    return individual


# Mutáció alkalmazása
def apply_mutation(population, mutation_rate, step_size):
    return np.array([mutate(ind, mutation_rate, step_size) for ind in population])


def relative_selection(population, fitness_values, num_selected):
    total_fitness = sum(fitness_values)
    selection_probs = [f / total_fitness for f in fitness_values]
    selected_indices = np.random.choice(len(population), num_selected, replace=False, p=selection_probs)
    return population[selected_indices]


def roulette_wheel_selection(population, fitness_values, num_selected):
    total_fitness = sum(fitness_values)
    selection_probs = [f / total_fitness for f in fitness_values]
    selected_indices = np.random.choice(len(population), num_selected, replace=False, p=selection_probs)
    return population[selected_indices]


def tournament_selection(population, fitness_values, num_selected, tournament_size=3):
    selected_indices = []
    for _ in range(num_selected):
        participants = np.random.choice(len(population), tournament_size, replace=False)
        best_participant = min(participants, key=lambda idx: fitness_values[idx])
        selected_indices.append(best_participant)
    return population[selected_indices]


# n dimenziohoz def-k
def one_point_crossover(parent1, parent2):
    point = np.random.randint(1, len(parent1))
    offspring1 = np.concatenate([parent1[:point], parent2[point:]])
    offspring2 = np.concatenate([parent2[:point], parent1[point:]])
    return offspring1, offspring2


def two_point_crossover(parent1, parent2):
    point1 = np.random.randint(1, len(parent1) - 1)
    point2 = np.random.randint(point1 + 1, len(parent1))
    offspring1 = np.concatenate([parent1[:point1], parent2[point1:point2], parent1[point2:]])
    offspring2 = np.concatenate([parent2[:point1], parent1[point1:point2], parent2[point2:]])
    return offspring1, offspring2


def multi_point_crossover(parent1, parent2, points=3):
    indices = sorted(np.random.choice(range(1, len(parent1)), points, replace=False))
    offspring1, offspring2 = parent1.copy(), parent2.copy()
    for i in range(0, len(indices), 2):
        start = indices[i]
        end = indices[i + 1] if i + 1 < len(indices) else len(parent1)
        offspring1[start:end], offspring2[start:end] = offspring2[start:end], offspring1[start:end]
    return offspring1, offspring2


def uniform_crossover(parent1, parent2):
    mask = np.random.randint(0, 2, size=len(parent1))
    offspring1 = np.where(mask, parent1, parent2)
    offspring2 = np.where(mask, parent2, parent1)
    return offspring1, offspring2


def path_relinking_crossover(parent1, parent2):
    offspring1 = parent1.copy()
    for i in range(len(parent1)):
        if np.random.rand() < 0.5:
            step = (parent2[i] - parent1[i]) * np.random.rand()
            offspring1[i] += step
    offspring2 = parent2.copy()
    for i in range(len(parent2)):
        if np.random.rand() < 0.5:
            step = (parent1[i] - parent2[i]) * np.random.rand()
            offspring2[i] += step
    return offspring1, offspring2


def run_genetic_algorithm(function, generations, population_size, step_size, num_selected, mutation_rate, selection_method, elitism_level=0, no_improve_gen=15):
    population = initialize_population(population_size, 2, bounds)
    best_fitness = np.inf
    no_improve_count = 0

    for gen in range(generations):
        fitness_values = [fitness_function(individual, function) for individual in population]

        # Elitizmus: Legjobb egyed(ek) kiválasztása
        elite_indices = np.argsort(fitness_values)[:elitism_level]
        elite_individuals = population[elite_indices]

        # Kiválasztás és utódok létrehozása
        if selection_method == "roulette":
            selected_population = roulette_wheel_selection(population, fitness_values, num_selected)
        elif selection_method == "tournament":
            selected_population = tournament_selection(population, fitness_values, num_selected)
        else:
            selected_population = rank_selection(population, fitness_values, num_selected)

        offspring_population = create_offspring(selected_population, population_size - num_selected - elitism_level)
        population = np.concatenate([elite_individuals, selected_population, offspring_population])
        population = apply_mutation(population, mutation_rate, step_size)

        current_best = np.min(fitness_values)
        if current_best < best_fitness:
            best_fitness = current_best
            no_improve_count = 0
        else:
            no_improve_count += 1

        print(f"Generáció {gen + 1}: Legjobb fitness = {best_fitness}")

        if no_improve_count >= no_improve_gen:
            break

    return best_fitness, np.max([fitness_function(ind, function) for ind in population]), np.std([fitness_function(ind, function) for ind in population])


# Inicializálja a results_df DataFrame-et
results_df = pd.DataFrame(columns=["Function", "Generations", "Population Size", "Step Size",
                                   "Best Fitness", "Worst Fitness", "Standard Deviation"])

functions = [rastrigin_modified, booth, levi]
generations_list = [5, 10, 20, 50, 100]
population_sizes = [5, 10, 20, 50, 100]
step_sizes = [0.1, 0.2, 0.5, 1, 1.5, 2]
elitism_levels = [0, 1, 2]  # Választható elitizmus szintek

results = []  # Eredményeket tároló lista

# Futassa az algoritmust a különböző kiválasztási módszerekkel és mentse az eredményeket külön CSV fájlokba
for selection_method in ["roulette", "tournament"]:
    for elitism_level in elitism_levels:
        results = []
        for function in functions:
            for generations in generations_list:
                for population_size in population_sizes:
                    for step_size in step_sizes:
                        start_time = time.time()
                        num_selected = population_size // 2
                        mutation_rate = 0.1
                        best_fitness, worst_fitness, std_dev = run_genetic_algorithm(
                            function, generations, population_size, step_size, num_selected, mutation_rate,
                            selection_method, elitism_level=elitism_level)
                        elapsed_time = time.time() - start_time
                        results.append({
                            "Function": function.__name__,
                            "Generations": generations,
                            "Population Size": population_size,
                            "Step Size": step_size,
                            "Best Fitness": best_fitness,
                            "Worst Fitness": worst_fitness,
                            "Standard Deviation": std_dev,
                            "Elapsed Time": elapsed_time
                        })

        results_df = pd.DataFrame(results)
        filename = f"outputFiles/genetic_algorithm_results_{selection_method}_elitism{elitism_level}.csv"
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(results_df.columns)
            for index, row in results_df.iterrows():
                writer.writerow(row)


# Különböző dimenziók és keresztezési módszerek futtatása
dimensions = [3, 4, 5, 10, 100]
crossover_methods = [one_point_crossover, two_point_crossover, multi_point_crossover, uniform_crossover, path_relinking_crossover]

for dim in dimensions:
    for crossover_method in crossover_methods:
        for elitism_level in elitism_levels:
            results = []
            for generations in generations_list:
                for population_size in population_sizes:
                    for step_size in step_sizes:
                        start_time = time.time()
                        population = initialize_population(population_size, dim, bounds)
                        best_fitness, worst_fitness, std_dev = run_genetic_algorithm(
                            n_dim_rastrigin, generations, population_size, step_size, population_size // 2, 0.1,
                            crossover_method, elitism_level=elitism_level)
                        elapsed_time = time.time() - start_time
                        results.append({
                            "Dimension": dim,
                            "Crossover Method": crossover_method.__name__,
                            "Generations": generations,
                            "Population Size": population_size,
                            "Step Size": step_size,
                            "Best Fitness": best_fitness,
                            "Worst Fitness": worst_fitness,
                            "Standard Deviation": std_dev,
                            "Elapsed Time": elapsed_time
                        })

            results_df = pd.DataFrame(results)
            filename = f"outputFiles/rastrigin_results_dim{dim}_{crossover_method.__name__}_elitism{elitism_level}.csv"
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(results_df.columns)
                for index, row in results_df.iterrows():
                    writer.writerow(row)


# Itt ér véget az algoritmus, és itt mérjük meg az eltelt időt
global_end_time = time.time()
elapsed_time = global_end_time - global_start_time
print(f"Az algoritmus teljes futási ideje: {elapsed_time} másodperc")
