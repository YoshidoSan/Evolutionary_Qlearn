# Authors: Kacper Marchlewicz, Przemysław Wyziński

from cec2017.functions import f4
from model import Model
from statistics import mean
import numpy as np
from qlearn import Qlearning

"""
Znajdywanie odpowiednich parametrów dla algorytmów, którymi q learning się nie zajmuje
założenia: 10000 iteracji, cec4, cec5 w 10 wymiarach, ograniczenia kostkowe 100 (jak na wsi)
szukamy akceptowalnych wyników dla liczby populacji, siły mutacji i rozmiaru elity
25 uruchomień, min, max, średnia, odchylenie
prawd krzyżowania na 0.5 uśreniające
na koniec szukanie crossover_parameter dla krzyżowania wymieniającego
domyśle wartośći : populacja 20, siła mutacji 1, rozmiar elity 1, prawd krzyżowania 0.5
"""


def calculate_params_population(function_calls, dimension):
    population_size = [5, 10, 15, 20, 30, 40, 100]
    scores = []
    for population in population_size:
        i = 0
        max_epoch = round(function_calls / population)
        while i < 25:
            model = Model(
                population_size=population, specimen_dimension=dimension, optim_function=f4)
            for epoch in range(1, max_epoch + 1):
                model.torunament_selection()
                model.crossover_intermediate(0.5)
                model.mutation(1)
                model.succession(1)
                model.set_best_specimen()
                spec, score = model.get_best_specimen()
                better_children = round(model.calculate_better_offsprings(), 2)
                distance = round(model.calculate_avg_distance(), 2)
            scores.append(score)
            i += 1
        print(
            f'Training ended, param_value: {population}, better children: {better_children}%, Avg distance: {distance}, Score: {score}')
        print(f'Population checked')
        print("Maximum: ", "%.2f" % max(scores))
        print("Minimum: ", "%.2f" % min(scores))
        print("Mean: ", "%.2f" % mean(scores))
        print("Std: ", "%.2f" % np.std(scores))


def calculate_params_mutation(function_calls, dimension):
    mutation_strength = [0.05, 0.1, 0.5, 0.75, 1, 2, 5, 10]
    population = 20
    max_epoch = round(function_calls / population)
    scores = []
    for mutation in mutation_strength:
        i = 0
        while i < 25:
            model = Model(
                population_size=population, specimen_dimension=dimension, optim_function=f4)
            for epoch in range(1, max_epoch + 1):
                model.torunament_selection()
                model.crossover_intermediate(0.5)
                model.mutation(mutation)
                model.succession(1)
                model.set_best_specimen()
                spec, score = model.get_best_specimen()
                better_children = round(model.calculate_better_offsprings(), 2)
                distance = round(model.calculate_avg_distance(), 2)
            scores.append(score)
            i += 1
        print(
            f'Training ended, param_value: {mutation}, better children: {better_children}%, Avg distance: {distance}, Score: {score}')
        print(f'Mutation checked')
        print("Maximum: ", "%.2f" % max(scores))
        print("Minimum: ", "%.2f" % min(scores))
        print("Mean: ", "%.2f" % mean(scores))
        print("Std: ", "%.2f" % np.std(scores))


def calculate_params_elite(function_calls, dimension):
    elite_size = [0, 1, 2, 3, 4]
    population = 20
    max_epoch = round(function_calls / population)
    scores = []
    for elite in elite_size:
        i = 0
        while i < 25:
            model = Model(
                population_size=population, specimen_dimension=dimension, optim_function=f4)
            for epoch in range(1, max_epoch + 1):
                model.torunament_selection()
                model.crossover_intermediate(0.5)
                model.mutation(1)
                model.succession(elite)
                model.set_best_specimen()
                spec, score = model.get_best_specimen()
                better_children = round(model.calculate_better_offsprings(), 2)
                distance = round(model.calculate_avg_distance(), 2)
            scores.append(score)
            i += 1
        print(
            f'Training ended, param_value: {elite}, better children: {better_children}%, Avg distance: {distance}, Score: {score}')
        print(f'Elite size checked')
        print("Maximum: ", "%.2f" % max(scores))
        print("Minimum: ", "%.2f" % min(scores))
        print("Mean: ", "%.2f" % mean(scores))
        print("Std: ", "%.2f" % np.std(scores))


def calculate_params_crossover_prop(function_calls, dimension):
    crossover_prop = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    population = 20
    max_epoch = round(function_calls / population)
    scores = []
    for cross in crossover_prop:
        i = 0
        while i < 25:
            model = Model(
                population_size=population, specimen_dimension=dimension, optim_function=f4)
            for epoch in range(1, max_epoch + 1):
                model.torunament_selection()
                model.crossover_intermediate(cross)
                model.mutation(1)
                model.succession(1)
                model.set_best_specimen()
                spec, score = model.get_best_specimen()
                better_children = round(model.calculate_better_offsprings(), 2)
                distance = round(model.calculate_avg_distance(), 2)
            scores.append(score)
            i += 1
        print(
            f'Training ended, param_value: {cross}, better children: {better_children}%, Avg distance: {distance}, Score: {score}')
        print(f'Crossover propability checked')
        print("Maximum: ", "%.2f" % max(scores))
        print("Minimum: ", "%.2f" % min(scores))
        print("Mean: ", "%.2f" % mean(scores))
        print("Std: ", "%.2f" % np.std(scores))


if __name__ == "__main__":
    calculate_params_population(function_calls=10000, dimension=10)
    calculate_params_elite(function_calls=10000, dimension=10)
    calculate_params_mutation(function_calls=10000, dimension=10)
    calculate_params_crossover_prop(function_calls=10000, dimension=10)
