# Authors: Kacper Marchlewicz, Przemysław Wyziński

import random
import numpy as np
import math
from statistics import mean

"""
akcje na q mapie:
0 -> zmiana sposobu krzyżowania
1 -> zostawienie sposobu krzyżowania
2 -> zwiększenie prawdopodobieństwa krzyżowania
3 -> zmniejszenie prawdopodobieństwa krzyżowania
4 -> zostawienie prawdopodobieństwa krzyżowania
"""

class Qlearning:
    def __init__(self, beta, gamma, e,  evolution_model, mutation_strength, elite_size, crossover_prop, crossover_type, function_calls,
                 crossover_param=0.0):
        self.evolution_model = evolution_model
        self.mutation_strength = mutation_strength
        self.elite_size = elite_size
        self.crossover_prop = crossover_prop
        self.crossover_type = crossover_type
        self.crossover_param = crossover_param
        self.function_calls = function_calls
        self.init_q_table()
        # parametry q learningu -> na podstawie badań
        self.beta = beta
        self.gamma = gamma
        self.e = e
        self.trained_state = None

    def init_q_table(self):
        self.q_rows = 15
        self.q_columns = 23
        self.q_table = np.zeros((self.q_rows, self.q_columns, 5))

    def percentage_discretization(self, children_precentages) -> int:
        baskets = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
        if children_precentages == 0:
            return 0
        for i in range(0, len(baskets) - 2):
            lower = baskets[i]
            upper = baskets[i + 1]
            if lower < children_precentages <= upper:
                return i

    def distance_discretization(self, avg_distance) -> int:
        baskets = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300,
                   math.inf]
        if avg_distance == 0:
            return 0
        for i in range(0, len(baskets) - 2):
            lower = baskets[i]
            upper = baskets[i + 1]
            if lower < avg_distance <= upper:
                return i

    def take_action(self, action):
        if action == 0:
            self.crossover_type = (-1) * self.crossover_type
            return 1
        if action == 1:
            return 1
        if action == 2:
            if self.crossover_prop >= 1:
                self.crossover_prop = 1
                return -1
            else:
                self.crossover_prop += 0.1
                return 1
        if action == 3:
            if self.crossover_prop <= 0:
                self.crossover_prop = 0
                return -1
            else:
                self.crossover_prop -= 0.1
                return 1
        if action == 4:
            return 1

    def get_reward(self, best_old_pop, new_pop):
        count = 0
        new_scores = self.evolution_model._optim_function(new_pop)
        for new in new_scores:
            if new > best_old_pop:
                count +=1
        count = round((count*100)/len(new_pop), 2)
        return count

    def learn(self, max_episodes):
        # w nowy epizod wejdziemy gdy ewolucyjny osiągnie koniec
        for ep in range(1, max_episodes + 1):
            finish = False
            current_state_1 = random.randint(0, 14)
            current_state_2 = random.randint(0, 22)
            # koniec gdy ewolucyjny zakończy działanie
            while finish is False:
                self.evolution_model.generate_population()
                max_epoch = round(self.function_calls / self.evolution_model._size)
                # uruchamiamy ewolucyjny
                for epoch in range(1, max_epoch + 1):
                    # pętla dopóki wybrana akcja będzie możliwa do wykonania (-1 oznacza brak możliwości wykonania ruchu)
                    can_do_action = -1
                    while can_do_action == -1:
                        # strategia zachłanna
                        number = random.uniform(0, 1)
                        # random action
                        if number <= self.e:
                            action = random.randint(0, 4)
                        # choose action from q table
                        else:
                            # boierzemy nagrody ze stanu
                            state_rewards = self.q_table[current_state_1][current_state_2]
                            # id max nagrody w stanie
                            max_vad_ids = np.where(state_rewards == max(self.q_table[current_state_1][current_state_2]))
                            # id max nagrody daje id akcji
                            action = np.random.choice(max_vad_ids[0].tolist(), 1)
                            action = action[0]
                        can_do_action = self.take_action(action)
                    # wykonanie akcji
                    self.evolution_model.torunament_selection()
                    if self.crossover_type == 1:
                        self.evolution_model.crossover_intermediate(self.crossover_prop)
                    else:
                        self.evolution_model.crossover_exchange(self.crossover_prop, self.crossover_param)
                    self.evolution_model.mutation(self.mutation_strength)
                    self.evolution_model.set_best_specimen()
                    parent_spec, parent_score = self.evolution_model.get_best_specimen()
                    score = self.get_reward(parent_score, self.evolution_model.get_next_population())
                    self.evolution_model.succession(self.elite_size)
                    self.evolution_model.set_best_specimen()
                    next_gen_spec, next_gen_score = self.evolution_model.get_best_specimen()
                    better_children = self.evolution_model.calculate_better_offsprings()
                    distance = self.evolution_model.calculate_avg_distance()
                    next_state_1 = self.percentage_discretization(better_children)
                    next_state_2 = self.distance_discretization(distance)
                    # nagroda -> procent lepszych dzieci
                    reward = score
                    # calculate q table values
                    delta = reward + self.gamma * np.max(self.q_table[next_state_1][next_state_2]) - \
                            self.q_table[current_state_1][current_state_2][action]
                    self.q_table[current_state_1][current_state_2][action] = \
                    self.q_table[current_state_1][current_state_2][action] + self.beta * delta
                    current_state_1 = next_state_1
                    current_state_2 = next_state_2
                finish = True
            print(f'Training Episode: {ep}/{max_episodes}, final point: {round(next_gen_score,2)}, alg last score: {round(reward,2)}')
        self.trained_state = True

    def test(self, show_epoch_info=False):
        if self.trained_state is None:
            raise ValueError('Model not trained yet')
        i = 0
        scores = []
        while i < 25:
            self.crossover_prop = 0.4
            current_state_1 = random.randint(0, 14)
            current_state_2 = random.randint(0, 22)
            self.evolution_model.generate_population()
            max_epoch = round(self.function_calls / self.evolution_model._size)
            for epoch in range(1, max_epoch + 1):
                # choose action from q table
                state_rewards = self.q_table[current_state_1][current_state_2]
                # id max nagrody w stanie
                max_vad_ids = np.where(state_rewards == max(self.q_table[current_state_1][current_state_2]))
                # id max nagrody daje id akcji
                action = np.random.choice(max_vad_ids[0].tolist(), 1)
                action = action[0]
                can_do_action = self.take_action(action)
                # wykonanie akcji
                self.evolution_model.torunament_selection()
                if self.crossover_type == 1:
                    self.evolution_model.crossover_intermediate(self.crossover_prop)
                else:
                    self.evolution_model.crossover_exchange(self.crossover_prop, self.crossover_param)
                self.evolution_model.mutation(self.mutation_strength)
                self.evolution_model.succession(self.elite_size)
                self.evolution_model.set_best_specimen()
                spec, score = self.evolution_model.get_best_specimen()
                better_children = self.evolution_model.calculate_better_offsprings()
                distance = self.evolution_model.calculate_avg_distance()
                next_state_1 = self.percentage_discretization(better_children)
                next_state_2 = self.distance_discretization(distance)
                current_state_1 = next_state_1
                current_state_2 = next_state_2
                if show_epoch_info:
                    crossover_name = 'intermediate' if self.crossover_type == 1 else 'exchange'
                    print(f'Epoch {epoch}/{max_epoch}, best specimen score: {round(score,2)}, better children: {round(better_children,2)}%, '
                          f'Avg distance: {round(distance,2)}, crossover propability: {round(self.crossover_prop,2)}, crossover '
                          f'type: {crossover_name}')
            i+=1
            scores.append(score)
            print(f'Run ended, best specimen score: {score}, final crossover propability: {self.crossover_prop}, final '
                  f'crossover type: {crossover_name}')
        print(f'Q-learn finished running')
        print("Maximum: ", "%.2f" % max(scores))
        print("Minimum: ", "%.2f" % min(scores))
        print("Mean: ", "%.2f" % mean(scores))
        print("Std: ", "%.2f" % np.std(scores))
        a = max(scores)
        b = min(scores)
        c = mean(scores)
        d = np.std(scores)
        return a, b, c, d
