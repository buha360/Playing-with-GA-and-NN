import matplotlib.pyplot as plt
import random


# TSP Fitness függvény
def fitness_tsp(distances, route):
    dist = 0
    prev = route[0]
    for i in route[1:]:
        dist += distances[(prev, i)]
        prev = i
    dist += distances[(route[-1], route[0])]
    return dist


# TSP populáció inicializálása
def initialize_tsp_population(population_size, num_cities):
    population = []
    for _ in range(population_size):
        route = list(range(num_cities))
        random.shuffle(route)
        population.append(route)
    return population


# TSP mutáció
def mutate_tsp(route, mutation_rate):
    for i in range(len(route)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(route) - 1)
            route[i], route[j] = route[j], route[i]
    return route


# TSP keresztezés
def crossover_tsp(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    offspring = [None] * len(parent1)
    offspring[start:end] = parent1[start:end]

    parent2_idx = 0
    for i in range(len(offspring)):
        if offspring[i] is None:
            while parent2[parent2_idx] in offspring:
                parent2_idx += 1
            offspring[i] = parent2[parent2_idx]
    return offspring


# A genetikus algoritmus futtatása a TSP-hez
def run_genetic_algorithm_tsp(distances, generations, population_size, mutation_rate):
    num_cities = max(max(city_pair) for city_pair in distances.keys()) + 1
    population = initialize_tsp_population(population_size, num_cities)
    best_route = None
    best_distance = float('inf')

    for gen in range(generations):
        # Fitness értékek kiszámítása
        fitness_values = [fitness_tsp(distances, route) for route in population]

        # Szelekció
        sorted_population = sorted(zip(population, fitness_values), key=lambda x: x[1])
        population = [route for route, _ in sorted_population[:population_size // 2]]

        # Keresztezés és mutáció
        offspring = []
        while len(offspring) < population_size - len(population):
            parent1, parent2 = random.sample(population, 2)
            child = crossover_tsp(parent1, parent2)
            child = mutate_tsp(child, mutation_rate)
            offspring.append(child)
        population += offspring

        # Legjobb útvonal frissítése
        current_best_distance = sorted_population[0][1]
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_route = sorted_population[0][0]

        # Logolás
        print(f"Generáció {gen + 1}: Legjobb távolság = {best_distance}")

    return best_route, best_distance


# Távolságok kiszámítása a koordináták alapján
def calculate_distance(coord1, coord2):
    return ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)**0.5


def plot_route(route, city_coords):
    x = [city_coords[i][0] for i in route]
    y = [city_coords[i][1] for i in route]

    # Az utolsó városhoz visszatérés
    x.append(city_coords[route[0]][0])
    y.append(city_coords[route[0]][1])

    plt.plot(x, y, 'o-', mfc='r')
    plt.xlabel('X koordináta')
    plt.ylabel('Y koordináta')
    plt.title('TSP Legjobb Útvonal')
    plt.show()


# Példa távolságok generálása
num_cities = 100
distances = {}
for i in range(num_cities):
    for j in range(num_cities):
        if i != j:
            if (j, i) not in distances:  # Ellenőrizzük, hogy a fordított irányú út már szerepel-e
                distances[(i, j)] = random.randint(1, 100)
                distances[(j, i)] = distances[(i, j)]  # Ugyanaz a távolság mindkét irányban


# Véletlenszerű koordináták generálása a városoknak
city_coords = {i: (random.uniform(0, 100), random.uniform(0, 100)) for i in range(num_cities)}

distances = {}
for i in range(num_cities):
    for j in range(num_cities):
        if i != j:
            distances[(i, j)] = calculate_distance(city_coords[i], city_coords[j])


# Algoritmus futtatása
best_route, best_distance = run_genetic_algorithm_tsp(distances, 2000, 200, 0.01)

print("Legjobb útvonal:", best_route)
print("Legjobb távolság:", best_distance)

plot_route(best_route, city_coords)
