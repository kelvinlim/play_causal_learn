import numpy as np
from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from causallearn.graph.ArrowConfusion import ArrowConfusion
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.PermutationBased.BOSS import boss
from causallearn.utils.DAG2CPDAG import dag2cpdag

# for FCI
from causallearn.search.ConstraintBased.FCI import fci, ruleR5
from causallearn.utils.cit import chisq, fisherz, kci, d_separation

import gc
import pandas as pd

import numpy as np
from numpy.linalg import LinAlgError

def run_boss_with_retry(X, lambda_value=4, retries=3, jitter=1e-8, reg=1e-8):
    for attempt in range(retries):
        try:
            return boss(X, parameters={'lambda_value': lambda_value})
        except LinAlgError as e:
            if attempt == retries - 1:
                raise
            # Try small jitter
            X = X + jitter * np.random.randn(*X.shape)
            jitter *= 10
    # last attempt (should have raised if it still fails)
    return boss(X, parameters={'lambda_value': lambda_value})

def run_fci_with_retry(X, lambda_value=4, retries=3, jitter=1e-8, reg=1e-8):
    for attempt in range(retries):
        try:
            return fci(X)
        except ValueError as e:
            if attempt == retries - 1:
                raise
            # Try small jitter
            X = X + jitter * np.random.randn(*X.shape)
            jitter *= 10
    # last attempt (should have raised if it still fails)
    return fci(X)

files = [
    "sim_data_renamed/sub_case-001_es-1.0_rows-100.csv",
    "sim_data_renamed/sub_case-001_es-1.0_rows-500.csv",
    "sim_data_renamed/sub_case-001_es-1.0_rows-1000.csv"


]

for file in files:
    # read in data into df
    df = pd.read_csv(file)
    # convert to numpy array
    X = df.to_numpy()

    # add jitter to data to avoid numerical issues
    eps = 1e-8
    X_jitter = X + eps * np.random.randn(*X.shape)
    #G = boss(X_jitter, parameters={'lambda_value': 4})
    G = run_boss_with_retry(X)
    gc.collect()
    print("BOSS Result for file:", file)
    print(G)
    
    #G_fci, edges = fci(X_jitter)
    G_fci, edges = run_fci_with_retry(X)
    #print("FCI Result for file:", file)
    #print(edges)
    
    
    # create plot
    # visualization
    from causallearn.utils.GraphUtils import GraphUtils
    pdy = GraphUtils.to_pydot(G_fci)
    # get basename of file using Path
    from pathlib import Path
    basename = Path(file).stem
    # save plot
    pdy.write_png(f'{basename}_fci_test.png')
    pass