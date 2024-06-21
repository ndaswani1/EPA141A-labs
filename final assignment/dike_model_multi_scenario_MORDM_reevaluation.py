# import libraries
from ema_workbench import (
    Model,
    MultiprocessingEvaluator,
    ScalarOutcome,
    IntegerParameter,
    optimize,
    Scenario,
    save_results,
    load_results,
    ema_logging,
    Policy
)
from ema_workbench.analysis import parcoords
from ema_workbench.em_framework.optimization import (ArchiveLogger, EpsilonProgress, rebuild_platypus_population, to_problem, epsilon_nondominated)
from problem_formulation import get_model_for_problem_formulation
from concurrent.futures import ProcessPoolExecutor
import os
import functools
from scipy.spatial.distance import pdist, squareform
import random
from scenario_selection import find_maxdiverse_scenarios
import numpy as np
import matplotlib as mpl

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import pandas as pd
import itertools

if __name__ == "__main__":
    # Function to analyze solutions across scenarios
    def analyse_solutions_across_scenarios(optimizer='opt200', seed_nr=3, scenario='reference'):
        """
        Analyzes the solutions generated by the optimization across all scenarios.

        Returns: pareto set for a scenario
        """
        # define model configurations
        epsilons = [1e4, 0.01, 1e4, 0.01, 1e4, 0.01, 1e4, 0.01, 1e4, 0.01, 1e4, 1e4]
        model, steps = get_model_for_problem_formulation(3)
        problem = to_problem(model, searchover="levers")


        # initiate a list of all archives
        all_archives = []
        for seed in range(seed_nr):
            # import data through archive logger
            archive = ArchiveLogger.load_archives(f"./results/{optimizer}/optimization_3_{scenario}_seed_{seed}.tar.gz")
            archive = archive[max(archive.keys())]
            archive = archive.iloc[:, 1:(len(archive.columns))]
            all_archives.append(archive)

        # compute the final pareto set for each scenario
        final_pareto_set = epsilon_nondominated(all_archives, epsilons, problem)

        # return pareto set
        return final_pareto_set


    # List of scenarios to analyze
    scenarios = ["reference", "67", "182", "257", "264"]
    all_results = []

    # Analyze solutions for each scenario and store the Pareto set results
    for scenario in scenarios:
        scenario_pareto_set = analyse_solutions_across_scenarios(optimizer='opt200', seed_nr=3, scenario=scenario)
        all_results.append(scenario_pareto_set)

    # Prepare data for visualization from the second scenario's results
    data = all_results[1].loc[:,
           ['A.1 Total Costs', 'A.1_Expected Number of Deaths', 'A.2 Total Costs', 'A.2_Expected Number of Deaths',
            'A.3 Total Costs', 'A.3_Expected Number of Deaths', 'RfR Total Costs', 'Expected Evacuation Costs']]


    # Determine limits for parallel coordinates plot axes
    limits = parcoords.get_limits(data)
    paraxes = parcoords.ParallelAxes(limits)
    colors = iter(sns.color_palette())
    for scenario_nr, result in enumerate(all_results):
        color = next(colors)
        data = result.loc[:,
               ['A.1 Total Costs', 'A.1_Expected Number of Deaths', 'A.2 Total Costs', 'A.2_Expected Number of Deaths',
                'A.3 Total Costs', 'A.3_Expected Number of Deaths', 'RfR Total Costs', 'Expected Evacuation Costs']]
        paraxes.plot(data, label=f'scenario {scenario_nr}', color=color)
    paraxes.legend()
    plt.show()

    policies = []
    for scenario_nr, scenario_result in enumerate(all_results):
        result = scenario_result.loc[:, ['A.1 Total Costs', 'A.1_Expected Number of Deaths', 'A.2 Total Costs',
                                         'A.2_Expected Number of Deaths', 'A.3 Total Costs',
                                         'A.3_Expected Number of Deaths', 'RfR Total Costs',
                                         'Expected Evacuation Costs']]
        for j, row in result.iterrows():
            policy = Policy(f'scenario {scenario_nr} option {j}', **row.to_dict())
            policies.append(policy)

    print("The preliminary total number of policies is now:", len(policies))

    scenarios = ["reference", "67", "182", "257", "264"]




    # Iterate over each scenario's result to apply constraints and create sliced Pareto sets
    sliced_all_pareto = []
    for i, result in enumerate(all_results):
        # define threshold
        result['All Costs'] = result[['A.1 Total Costs', 'A.2 Total Costs', 'A.3 Total Costs', 'RfR Total Costs',
                                      'Expected Evacuation Costs']].sum(axis=1)
        result['Dike Ring 1 - 3 Costs'] = result[['A.1 Total Costs', 'A.2 Total Costs', 'A.3 Total Costs']].sum(axis=1)
        result['A3ratio'] = result['A.3 Total Costs'] / result['Dike Ring 1 - 3 Costs']
        result['Dike Ring 1 - 5 Costs'] = result[['A.1 Total Costs', 'A.2 Total Costs', 'A.3 Total Costs',
                                                  'A.4 Total Costs', 'A.5 Total Costs']].sum(axis=1)
        result['ProvinceGratio'] = result['Dike Ring 1 - 3 Costs'] / result['Dike Ring 1 - 5 Costs']

        # constraint results
        result = result.loc[result['RfR Total Costs'] < 150e6]
        result = result.loc[result['A3ratio'] < 0.5]
        result = result.loc[result['ProvinceGratio'] < 0.5]

        # add sliced dataset to list
        sliced_all_pareto.append(result)

    policies = []
    # Plot each scenario's data on parallel axes with a label
    for scenario_nr, scenario_result in enumerate(sliced_all_pareto):
        levers = scenario_result.iloc[:, 0:(len(scenario_result.columns) - 12)]
        for j, row in levers.iterrows():
            policy = Policy(f'scenario {scenario_nr} option {j}', **row.to_dict())
            policies.append(policy)

    print("The total number of policies after brushing is:", len(policies))

    data = sliced_all_pareto[1].loc[:,
           ['A.1 Total Costs', 'A.1_Expected Number of Deaths', 'A.2 Total Costs', 'A.2_Expected Number of Deaths',
            'A.3 Total Costs', 'A.3_Expected Number of Deaths', 'RfR Total Costs', 'Expected Evacuation Costs']]

    limits = parcoords.get_limits(data)
    paraxes = parcoords.ParallelAxes(limits)
    colors = iter(sns.color_palette())


    for scenario_nr, scenario_result in enumerate(sliced_all_pareto):
        color = next(colors)
        data = scenario_result.loc[:, ['A.1 Total Costs', 'A.1_Expected Number of Deaths', 'A.2 Total Costs',
                                       'A.2_Expected Number of Deaths', 'A.3 Total Costs',
                                       'A.3_Expected Number of Deaths', 'RfR Total Costs',
                                       'Expected Evacuation Costs']]
        paraxes.plot(data, label=f'scenario {scenario_nr}', color=color)
    paraxes.legend()
    plt.show()

    #Reevaluation of policies under 1000 scenarios
    model, steps = get_model_for_problem_formulation(3)
    with MultiprocessingEvaluator(model) as evaluator:
        eval_results = evaluator.perform_experiments(1000, policies=policies)

    # save results
    experiments, outcomes = eval_results
    save_results((experiments, outcomes), './results/opt200/reevaluation_opt200.tar.gz')