# Authors: Kacper Marchlewicz, Przemysław Wyziński

import numpy as np
import random
import copy
import math

class Model():
    def __init__(self, population_size: int, specimen_dimension: int, optim_function, random_upperbound=100) -> None:
        self._size = population_size
        self._population = None
        self._families_parents_scores = []
        self._next_population = None
        self._families_children_scores = []
        self._best_specimen = None
        self._best_specimen_score = None
        self._optim_function = optim_function
        self._specimen_dimension = specimen_dimension
        self._random_upperbound = random_upperbound
        self.generate_population()

    def get_population(self) -> list:
        return self._population
    
    def get_next_population(self) -> list:
        return self._next_population

    def generate_population(self) -> None:
        self._population = np.random.uniform(
                -self._random_upperbound,
                self._random_upperbound,
                size=(self._size,self._specimen_dimension))

    def torunament_selection(self, show_logs=False) -> None:
        # rozmiar turnieju dwa -> jest to typowa wartość więc bym takie na stałe zostawił
        new_population = np.zeros([self._size,self._specimen_dimension])
        for i in range(self._size):
            specimens = np.array([random.choice(self._population), random.choice(self._population)])
            scores = self._optim_function(specimens)
            # zakładam zadanie minimalizacji
            winner = specimens[0] if scores[0] < scores[1] else specimens[1]
            if show_logs:
                print(f'--- Round {i+1} ---')
                print(f'Specimens: {specimens}\nwith scores: {scores}')
                print(f'Winner is: {winner}')
            for j in range(self._specimen_dimension):
                new_population[i,j] = winner[j]
        self._population = copy.deepcopy(new_population)

    def crossover_intermediate(self, cross_prob) -> None:
        # krzyżowanie uśredniające
        children_population = np.zeros([self._size, self._specimen_dimension])
        self._families_parents_scores = []
        for i in range(self._size):
            # losujemy dwóch rodziców
            id_1 = random.randint(0, len(self._population)-1)
            id_2 = random.randint(0, len(self._population)-1)
            while id_1 == id_2:
                id_2 = random.randint(0, len(self._population) - 1)
            parents = np.zeros([2, self._specimen_dimension])
            parents[0,:] = self._population[id_1]
            parents[1,:] = self._population[id_2]
            parents_scores = self._optim_function(parents)
            # zapis wyniku pary rodziców
            self._families_parents_scores.append([parents_scores[0], parents_scores[1]])
            # krzyżujemy - modyfikujemy punkty w potomku
            pr = random.uniform(0, 1)
            if pr < cross_prob:
                child = np.zeros([self._specimen_dimension])
                for k in range(self._specimen_dimension):
                    weight = random.uniform(0, 1)
                    child[k] = weight * parents[0,:][k] + (1-weight) * parents[1,:][k]
            # krzyżowanie nie udane - jeden z rodziców zostanie potomkiem
            else:
                pr_2 = random.uniform(0, 1)
                if pr_2 < 0.5:
                    child = parents[0,:]
                else:
                    child = parents[1,:]
            for j in range(self._specimen_dimension):
                children_population[i, j] = child[j]
        self._next_population = children_population

    def crossover_exchange(self, cross_prob, crossover_parameter) -> None:
        # krzyżowanie wymieniające
        children_population = np.zeros([self._size, self._specimen_dimension])
        self._families_parents_scores = []
        for i in range(self._size):
            # losujemy dwóch rodziców
            id_1 = random.randint(0, len(self._population)-1)
            id_2 = random.randint(0, len(self._population)-1)
            while id_1 == id_2:
                id_2 = random.randint(0, len(self._population) - 1)
            parents = np.zeros([2, self._specimen_dimension])
            parents[0,:] = self._population[id_1]
            parents[1,:] = self._population[id_2]
            parents_scores = self._optim_function(parents)
            # zapis wyniku pary rodziców
            self._families_parents_scores.append([parents_scores[0], parents_scores[1]])
            child = np.zeros([self._specimen_dimension])
            # krzyżujemy - modyfikujemy punkty w potomku
            pr = random.uniform(0, 1)
            if pr < cross_prob:
                for k in range(self._specimen_dimension):
                    number = random.uniform(0, 1)
                    if number < crossover_parameter:
                        child[k] = parents[0,k]
                    else:
                        child[k] = parents[1,k]
            # krzyżowanie nie udane - jeden z rodziców zostanie potomkiem
            else:
                pr_2 = random.uniform(0, 1)
                if pr_2 < 0.5:
                    child = parents[0,:]
                else:
                    child = parents[1,:]

            for j in range(self._specimen_dimension):
                children_population[i, j] = child[j]
        self._next_population = children_population
        
    def mutation(self, mutation_strength) -> None:
        # mutacja gaussowska
        self._families_children_scores = []
        i = 0
        while i < self._size:
            n = np.random.normal(0, 1, size=self._specimen_dimension)
            self._next_population[i] = self._next_population[i] + mutation_strength * n
            # zapis wyniku dziecka i-tej pary
            self._families_children_scores.append(self._optim_function([self._next_population[i]]))
            i += 1

    def succession(self, elite_size) -> None:
        # sukcesja elitarna -> potrzebne posortowane populacje
        # sortowanie starej populacji 
        scores = self._optim_function(self._population)
        sorted_pops, _ = self.sort_population(self._population, scores)
        self._population = copy.deepcopy(sorted_pops)
        # sortowanie nowej populacji
        scores = self._optim_function(self._next_population)
        sorted_pops, _ = self.sort_population(self._next_population, scores)
        self._population = copy.deepcopy(sorted_pops)
        i = 0
        # do nowej populacji przechodzą stare
        while i < elite_size:
            self._next_population[-1 - i] = self._population[i]
            i += 1
        # nowa populacja staje się rodzicami
        self._population = self._next_population

    def get_best_specimen(self) -> tuple[list, float]:
        return self._best_specimen, self._best_specimen_score

    def set_best_specimen(self):
        scores = self._optim_function(self._population)
        sorted_pops, sorted_scores = self.sort_population(self._population, scores)
        best_specimen = sorted_pops[0]
        best_specimen_score = sorted_scores[0]
        if self._best_specimen_score is None or self._best_specimen_score > best_specimen_score:
            self._best_specimen = best_specimen
            self._best_specimen_score = best_specimen_score

    def sort_population(self, pops_to_sort, scores_to_sort) -> tuple[list,list]:
        # sortuje populacje i wynik według oceny -> zakładam że minimalizacja
        leng = len(scores_to_sort)
        sorted_pops = []
        sorted_scores = []
        pops_to_sort = [pops_to_sort[i] for i in range(len(pops_to_sort))]
        scores_to_sort = [scores_to_sort[i] for i in range(len(scores_to_sort))]
        i = 0
        while i < leng:
            to_get = np.argmin(scores_to_sort)
            sorted_pops.append(pops_to_sort[to_get])
            sorted_scores.append(scores_to_sort[to_get])
            pops_to_sort.pop(to_get)
            scores_to_sort.pop(to_get)
            i += 1
        return np.array(sorted_pops), np.array(sorted_scores)

    def calculate_better_offsprings(self) -> float:
        counter = 0
        # idziemy po parach rodziców i ich dziecku
        for i in range(len(self._families_children_scores)):
            parents_best_score = min(self._families_parents_scores[i])
            if self._families_children_scores[i] < parents_best_score:
                counter += 1
        percentage = (counter / len(self._families_children_scores)) * 100
        return percentage

    def calculate_avg_distance(self) -> float:
        distance_sum = 0
        pair_number = 0
        # iteracja po wszystkich kombinacjach par punktów
        for i in range(len(self._population)):
            for j in range(i + 1, len(self._population)):
                # Obliczenie odległości euklidesowej między punktami
                distance = math.dist(self._population[i], self._population[j])
                distance_sum += distance
                pair_number += 1
        avg = distance_sum / pair_number
        return avg
