import sys 
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import pearsonr

# Append parent directory for local module imports if needed
sys.path.append("../")

from my_src import constants
from my_src.utils import save_results, normalise_data

from nsm_toolbox.src import network_diffusion_model
from nsm_toolbox.src.find_optimal_timepoint import find_optimal_timepoint



def run_ndm_for_participant(clinical_group, 
                        participant, 
                        t=np.arange(0, 50, 0.1),
                        gamma=1,
                        connectome_path=constants.connectome_path):
    
    region_list = constants.tau_region_raj_label
    connectome_path = connectome_path.format(clinical_group_name=clinical_group, participant=participant)
    
    tau = pd.read_csv(constants.tau_filepath, names=["region", "suvr"], header=0)
    target_data = tau["suvr"].values
    # remove the subcortical regions, since these are affected by off-target binding of the tau-PET tracer
    CORT_IDX = np.concatenate([np.arange(34), np.arange(49, 83)])
    target_data = target_data[CORT_IDX]
    target_data = normalise_data(target_data)
    
    model = network_diffusion_model.NDM(connectome_fname=connectome_path,  # set up our network diffusion model class
                             t=t,
                             gamma=gamma,
                             ref_list=region_list)
    
    df, optimal_seed = model.optimise_seed_region(target_data)
    model.seed_region = optimal_seed["seed"]
    model_output = model.run_NDM()
    min_idx, prediction, SSE = find_optimal_timepoint(model_output, target_data)
    r = pearsonr(prediction, target_data)[0]
    
    save_results(
        clinical_group_name=clinical_group,
        participant=participant,
        seed=model.seed_region,
        alpha=None,  # NDM doesn't use alpha parameter
        r=r,
        SSE=SSE,
        model_output=model_output,
        prediction=prediction,
        optimization_iters=df,  # Store the seed optimization dataframe as iters
        model="NDM"
    )
    
    return dict(
        optimal_parameters={"seed": model.seed_region},
        SSE=SSE,
        prediction=prediction,
        model_output=model_output,
        r=r
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NDM model across clinical groups and participant ranges")
    parser.add_argument("--start", type=int, required=True, help="Start participant index (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="End participant index (exclusive)")
    parser.add_argument("--include_average", type=bool, required=False, default=False, help="Include average connectome in the analysis")
    
    args = parser.parse_args()
    
    start_participant = args.start
    end_participant = args.end

    # Create a list of tasks for all clinical group and participant pairs.
    tasks = [
        delayed(run_ndm_for_participant)(group, participant)
        for group in constants.clinical_group_names
        for participant in range(start_participant, end_participant)
    ]
    if args.include_average:
        tasks += [
            delayed(run_ndm_for_participant)(group, "avg")
            for group in constants.clinical_group_names
        ]
    
    total_tasks = len(tasks)
    
    # Use tqdm to track progress
    results = Parallel(n_jobs=-1)(
        tqdm(tasks, total=total_tasks, desc="Running NDM")
    )