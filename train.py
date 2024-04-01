# Authors: Kacper Marchlewicz, Przemysław Wyziński

import numpy as np
from cec2017.functions import f6 as optfun
from model import Model
from qlearn import Qlearning
from datetime import datetime


def only_evolution_alg(population, dimension, function_calls, mutation_strength, elite_size, crossover_type,
                       crossover_prop, crossover_param=0):
    model = Model(
        population_size=population, specimen_dimension=dimension, optim_function=optfun)
    max_epoch = round(function_calls / population)
    for epoch in range(1, max_epoch + 1):
        model.torunament_selection()
        if crossover_type == 0:
            model.crossover_intermediate(crossover_prop)
        else:
            model.crossover_exchange(crossover_prop, crossover_param)
        model.mutation(mutation_strength)
        model.succession(elite_size)
        model.set_best_specimen()
        spec, score = model.get_best_specimen()
        better_children = model.calculate_better_offsprings()
        distance = model.calculate_avg_distance()
        print(f'Epoch {epoch}/{max_epoch}, best specimen: {spec} with score: {score}')
    print(f'Training ended, better children: {better_children}%, Avg distance: {distance}')


def calculate_q_learn_params():
    param_values = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1]
    q_learn_iter = 1
    values = []
    now = datetime.now()
    with open(f'q_learng_params_logs.txt', 'w') as fh:
        fh.write(f'Log for function {optfun.__name__} started at {now.strftime("%H:%M:%S")}\n')
    for val1 in param_values:
        beta = val1
        for val2 in param_values:
            print(f"################ Starting Q-learin optimalization {q_learn_iter}/{len(param_values)**2} iterations ################")
            gamma = val2
            print(f"beta: {beta}, gamma: {gamma}")
            a, b, c, d = train_loop(population=40, dimension=10, beta=beta, gamma=gamma, e=0.1, function_calls=10000,
                                    mutation_strength=1, elite_size=2, crossover_type=1,
                                    crossover_prop=0.5, crossover_param=0.3)
            run = []
            scores = [a, b, c, d]
            run.append(beta)
            run.append(gamma)
            run.append(scores)
            values.append(run)
            q_learn_iter += 1
            with open(f'q_learng_params_logs.txt', 'a') as fh:
                fh.write(str(run)+'\n')
    print(values)
    now = datetime.now()
    with open(f'q_learng_params_logs.txt', 'a') as fh:
        fh.write(f'Log ended at {now.strftime("%H:%M:%S")}\n')


def train_loop(population, dimension, beta, gamma, e, mutation_strength, elite_size, crossover_prop, crossover_type,
               function_calls, crossover_param=0):
    model = Model(
        population_size=population, specimen_dimension=dimension, optim_function=optfun)
    qlearn = Qlearning(beta=beta, gamma=gamma, e=e,
                       evolution_model=model, mutation_strength=mutation_strength, elite_size=elite_size,
                       function_calls=function_calls, crossover_type=crossover_type, crossover_prop=crossover_prop,
                       crossover_param=crossover_param)
    print('--- Start q-learning ---')
    qlearn.learn(max_episodes=200)
    print('--- Learning finished ---')
    print('--- Testing ---')
    a, b, c, d = qlearn.test(show_epoch_info=True)
    return a, b, c, d


if __name__ == "__main__":
    calculate_q_learn_params()
