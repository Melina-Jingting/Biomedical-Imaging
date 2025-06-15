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

from nsm_toolbox.src import FKPP_model
from nsm_toolbox.src.find_optimal_timepoint import find_optimal_timepoint


def run_wfkpp_permuted_amyloid(clinical_group,
                        participant,
                        permutation_id,
                        t=np.arange(0, 50, 0.1),
                        gamma=1,
                        n_calls=200,
                        seed_list=None,
                        connectome_filepath=constants.connectome_filepath):
    """
    Run the A-beta FKPP model with permuted amyloid-beta maps
    
    Args:
        clinical_group: Clinical group name
        participant: Participant identifier
        permutation_id: Identifier for this permutation (1-100)
        t: Time points for simulation
        gamma: Gamma parameter
        n_calls: Number of optimization calls
        seed_list: List of potential seed regions
        connectome_path: Path to connectome
    """
    region_list = constants.tau_region_raj_label
    connectome_filepath = connectome_filepath.format(clinical_group_name=clinical_group, participant=participant)
    
    # Load tau data
    tau = pd.read_csv(constants.tau_filepath, names=["region", "suvr"], header=0)
    target_data = tau["suvr"].values
    
    # Remove subcortical regions
    CORT_IDX = np.concatenate([np.arange(34), np.arange(49, 83)])
    target_data = target_data[CORT_IDX]
    target_data = normalise_data(target_data)
    
    # Load Amyloid-beta PET data
    amyloid_data = pd.read_csv(constants.amyloid_filepath, names=["region", "suvr"], header=0)
    amyloid_values = amyloid_data["suvr"].values.copy()
    
    # Permute amyloid-beta values - shuffle the values randomly
    np.random.seed(permutation_id)  # Set seed for reproducibility
    permuted_amyloid = amyloid_values.copy()
    np.random.shuffle(permuted_amyloid)  # This shuffles in-place
    
    # Initialize model with permuted amyloid weights
    model = FKPP_model.FKPP(connectome_fname=connectome_filepath,
                             t=t,
                             gamma=gamma,
                             ref_list=region_list,
                             weights=permuted_amyloid)  # Use permuted amyloid data as weights
    
    # Optimize model parameters
    optimization_iters, optimal_parameters = model.optimise_fkpp(target_data, n_calls=n_calls, seed_list=seed_list)
    model.seed_region = optimal_parameters["seed"]
    model.alpha = optimal_parameters["alpha"]
    model_output = model.run_FKPP()
    min_idx, prediction, SSE = find_optimal_timepoint(model_output, target_data)
    r = pearsonr(prediction, target_data)[0]
    
    # Save results with permutation ID in participant name
    participant_id = f"{participant}_perm{permutation_id:03d}"
    save_results(
        clinical_group_name=clinical_group,
        participant=participant_id,
        seed=model.seed_region,
        alpha=model.alpha,
        r=r,
        SSE=SSE,
        model_output=model_output,
        prediction=prediction,
        optimization_iters=optimization_iters,
        model="WFKPP_PERMUTE_AB"
    )
    
    return dict(
        permutation_id=permutation_id,
        optimal_parameters=optimal_parameters,
        SSE=SSE,
        r=r
    )

def run_permutation_analysis(clinical_groups, participants, n_permutations=100, n_calls=200, seed_list=None, n_jobs=-1):
    """
    Run permutation analysis on amyloid-beta maps.
    
    Args:
        clinical_groups: List of clinical groups
        participants: List of participants (can include "avg")
        n_permutations: Number of permutations to run
        n_calls: Number of optimization calls
        seed_list: List of potential seed regions
        n_jobs: Number of parallel jobs
    """
    # Create tasks for all clinical group, participant, and permutation combinations
    tasks = []
    for group in clinical_groups:
        for participant in participants:
            for perm_id in range(1, n_permutations + 1):
                tasks.append(
                    delayed(run_wfkpp_permuted_amyloid)(
                        group, participant, perm_id, seed_list=seed_list, n_calls=n_calls
                    )
                )
    
    total_tasks = len(tasks)
    
    # Run tasks in parallel with progress bar
    results = Parallel(n_jobs=n_jobs)(
        tqdm(tasks, total=total_tasks, desc="Running WFKPP permutation analysis")
    )
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WFKPP model with permuted amyloid-beta maps")
    parser.add_argument("--n_permutations", type=int, default=100, help="Number of permutations")
    parser.add_argument("--n_calls", type=int, default=200, help="Number of optimization calls")
    parser.add_argument("--constrain_seeds", action="store_true", help="Constrain seed regions to predefined list")
    parser.add_argument("--participant", type=str, default="avg", help="Participant to analyze (default: avg)")
    parser.add_argument("--clinical_group", type=str, default="CN", help="Clinical group to analyze (default: all groups)")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs (-1 for all cores)")
    
    args = parser.parse_args()
    
    # Determine seed list
    if args.constrain_seeds:
        seed_list = constants.wfkpp_seed_list
    else:
        seed_list = None
    
    # Determine clinical groups to process
    if args.clinical_group:
        clinical_groups = [args.clinical_group]
    else:
        clinical_groups = constants.clinical_group_names
    
    # Determine participants to process
    participants = [args.participant]
    
    # Run the permutation analysis
    results = run_permutation_analysis(
        clinical_groups=clinical_groups,
        participants=participants,
        n_permutations=args.n_permutations,
        n_calls=args.n_calls,
        seed_list=seed_list,
        n_jobs=args.n_jobs
    )
    
