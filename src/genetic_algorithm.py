import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

class GeneticAlgorithmFeatureSelection:
    def __init__(self, population_size=20, generations=10, mutation_rate=0.1):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
    def initialize_population(self, num_features):
        """
        Initialize a population of binary vectors (chromosomes)
        1: feature selected, 0: feature ignored
        """
        return np.random.randint(2, size=(self.population_size, num_features))
        
    def fitness_function(self, chromosome, X, y):
        """
        Evaluate the chromosome by passing only the selected features to a classifier
        and computing cross-validation accuracy.
        """
        # If no features selected, score is 0
        if np.sum(chromosome) == 0:
            return 0.0
            
        # Select active columns
        X_selected = X[:, chromosome == 1]
        
        # Use a lightweight classifier for fast evaluation
        clf = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42, n_jobs=-1)
        
        # Score via 3-fold cross validation
        scores = cross_val_score(clf, X_selected, y, cv=3, scoring='accuracy')
        
        # We can implement a penalty for too many features to encourage sparsity
        num_selected = np.sum(chromosome)
        penalty = 0.001 * num_selected
        
        return scores.mean() - penalty
        
    def selection(self, population, fitness_scores):
        """ Roulette wheel selection """
        # Shift fitness so minimum is 0
        fitness_scores = np.array(fitness_scores)
        if fitness_scores.min() < 0:
            fitness_scores = fitness_scores - fitness_scores.min()
            
        total_fitness = np.sum(fitness_scores)
        if total_fitness == 0:
            probabilities = np.ones(len(fitness_scores)) / len(fitness_scores)
        else:
            probabilities = fitness_scores / total_fitness
            
        selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
        return population[selected_indices]
        
    def crossover(self, parent1, parent2):
        """ Single point crossover """
        point = np.random.randint(1, len(parent1)-1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
        
    def mutation(self, chromosome):
        """ Bit flip mutation """
        for i in range(len(chromosome)):
            if np.random.rand() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i] # flip 0 to 1, or 1 to 0
        return chromosome
        
    def optimize(self, X, y):
        num_features = X.shape[1]
        population = self.initialize_population(num_features)
        
        best_overall_chromosome = None
        best_overall_fitness = -float('inf')
        
        print(f"Starting Genetic Algorithm: {self.generations} generations, Pop Size {self.population_size}")
        
        for gen in range(self.generations):
            # 1. Evaluate Fitness
            fitness_scores = [self.fitness_function(ind, X, y) for ind in population]
            
            # 2. Track best
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            best_ind = population[best_idx]
            
            if best_fitness > best_overall_fitness:
                best_overall_fitness = best_fitness
                best_overall_chromosome = best_ind.copy()
                
            print(f"Gen {gen+1}/{self.generations} | Best Fitness: {best_overall_fitness:.4f} | Features Selected: {np.sum(best_overall_chromosome)}/{num_features}")
            
            # 3. Selection
            parents = self.selection(population, fitness_scores)
            
            # 4. Crossover & Mutation
            next_generation = []
            for i in range(0, self.population_size, 2):
                p1 = parents[i]
                p2 = parents[(i+1) % self.population_size] # handle odd pop_size
                
                c1, c2 = self.crossover(p1, p2)
                
                c1 = self.mutation(c1)
                c2 = self.mutation(c2)
                
                next_generation.append(c1)
                next_generation.append(c2)
                
            population = np.array(next_generation)[:self.population_size]
            
        print("Optimization complete!")
        return best_overall_chromosome, best_overall_fitness
