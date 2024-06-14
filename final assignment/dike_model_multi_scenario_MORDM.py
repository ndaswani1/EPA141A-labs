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
    MPIEvaluator,
    ema_logging
)
from ema_workbench.em_framework.optimization import (ArchiveLogger, EpsilonProgress)

from problem_formulation import get_model_for_problem_formulation
from concurrent.futures import ProcessPoolExecutor
import os
import functools
from scipy.spatial.distance import pdist, squareform
import random
from scenario_selection import find_maxdiverse_scenarios
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import pandas as pd
import itertools

if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)
    # define the model and steps
    model, steps = get_model_for_problem_formulation(3)
    # retrieve scenarios and outcomes for scenario selection
    scenarios, outcomes = load_results("./scenarios_for_multiobj.tar.gz")
    # define which scenarios are of interest
    experiments_of_interest = scenarios['scenario']
    # select only outcomes for these experiments we are interested in
    outcomes_df = pd.DataFrame({k: v[experiments_of_interest] for k, v in outcomes.items()})
    # normalize outcomes on unit interval to ensure equal weighting of outcomes
    x = outcomes_df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalized_outcomes = pd.DataFrame(x_scaled, columns=outcomes_df.columns)
    # define which keys are uncertainties
    scenario_inputs = [i for i in scenarios.keys()]
    # retrieve only uncertainties from list
    scenario_inputs = scenario_inputs[0:19]
    # select columns which are uncertainties
    selected = scenarios[scenario_inputs]
    # most_diverse = ([1.699086703509649], np.array([ 67, 182, 257, 264])) # follows from analysis above
    indices = [67, 182, 257, 264]
    # select only these four scenarios
    selected = selected.loc[indices]
    # define EMA scenario
    scenarios = [Scenario(f"{index}", **row) for index, row in selected.iterrows()]
    # retrieve reference scenario
    reference_values = {
        "Bmax": 175,
        "Brate": 1.5,
        "pfail": 0.5,
        "discount rate 0": 3.5,
        "discount rate 1": 3.5,
        "discount rate 2": 3.5,
        "ID flood wave shape": 4,
    }
    scen1 = {}
    for key in model.uncertainties:
        name_split = key.name.split("_")
        if len(name_split) == 1:
            scen1.update({key.name: reference_values[key.name]})
        else:
            scen1.update({key.name: reference_values[name_split[1]]})
    # define reference scenario in EMA
    ref_scenario = Scenario("reference", **scen1)
    # add reference scenario to scenarios
    scenarios.append(ref_scenario)

    def optimize(model, scenario, nfe, epsilons, seed_nr):
        results = []
        convergences = []
        with MPIEvaluator(model) as evaluator:
            for i in range(seed_nr):
                convergence_metrics = [ArchiveLogger(
                    "./",
                    [l.name for l in model.levers],
                    [o.name for o in model.outcomes],
                    base_filename=f"optimization_3_{scenario.name}_seed_{i}.tar.gz"),
                    EpsilonProgress(),
                ]

                result, convergence = evaluator.optimize(nfe=nfe, searchover='levers',
                                                         convergence=convergence_metrics,
                                                         epsilons=epsilons,
                                                         reference=scenario)
                results.append(result)
                convergences.append(convergence)

        return results, convergences

    optimizations = []
    for scenario in scenarios:
        epsilons = [1e3, ] * len(model.outcomes)
        optimizations.append(optimize(model, scenario, 100000, epsilons, 3))