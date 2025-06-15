import sys
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from bct import null_model_und_sign

# Append parent directory for local module imports if needed
sys.path.append("../")

from my_src import constants
from my_src.utils import save_results, normalise_data, load_connectome

from nsm_toolbox.src import network_diffusion_model
from nsm_toolbox.src.find_optimal_timepoint import find_optimal_timepoint


def run_ndm_rewired(clinical_group,
                    permutation_id,
                    t=np.arange(0, 50, 0.1),
                    gamma=1,
                    bin_swaps=5,
                    wei_freq=0.1):
    """
    Run NDM model with rewired connectomes.
    
    Args:
        clinical_group: Clinical group name
        permutation_id: Identifier for this permutation (1-100)
        t: Time points for simulation
        gamma: Gamma parameter
        bin_swaps: Number of binary swaps for rewiring (default 5)
        wei_freq: Frequency of weight sorting in rewiring (default 0.1)
    """
    region_list = constants.tau_region_raj_label
    
    # Define the rewired connectome filepath
    rewired_connectome_filepath = constants.connectome_filepath.format(
        clinical_group_name="CN_REWIRED", 
        participant=f"avg_perm{permutation_id:03d}"
    )
    
    # Check if rewired connectome already exists
    if os.path.exists(rewired_connectome_filepath):
        print(f"Using existing rewired connectome: {rewired_connectome_filepath}")
        # No need to generate a new one, we'll use the existing file
        corr_metrics = None  # We don't have correlation metrics for existing files
    else:
        print(f"Generating new rewired connectome: {rewired_connectome_filepath}")
        # Make directory if it doesn't exist
        os.makedirs(os.path.dirname(rewired_connectome_filepath), exist_ok=True)
        
        # Load the original connectome
        original_connectome = load_connectome("CN", "avg")
        
        # Generate rewired connectome using null_model_und_sign
        np.random.seed(permutation_id)  # Set seed for reproducibility
        rewired_connectome, corr_metrics = null_model_und_sign(
            original_connectome, 
            bin_swaps=bin_swaps, 
            wei_freq=wei_freq, 
            seed=permutation_id
        )
        
        # Save the rewired connectome
        np.savetxt(rewired_connectome_filepath, rewired_connectome, delimiter=",")
    
    # Load tau data
    tau = pd.read_csv(constants.tau_filepath, names=["region", "suvr"], header=0)
    target_data = tau["suvr"].values
    
    # Remove subcortical regions if necessary
    CORT_IDX = np.concatenate([np.arange(34), np.arange(49, 83)])
    target_data = target_data[CORT_IDX]
    target_data = normalise_data(target_data)
    
    # Initialize and run NDM model with rewired connectome
    model = network_diffusion_model.NDM(
        connectome_fname=rewired_connectome_filepath,
        t=t,
        gamma=gamma,
        ref_list=region_list
    )
    
    # Optimize seed region
    df, optimal_seed = model.optimise_seed_region(target_data)
    model.seed_region = optimal_seed["seed"]
    model_output = model.run_NDM()
    min_idx, prediction, SSE = find_optimal_timepoint(model_output, target_data)
    r = pearsonr(prediction, target_data)[0]
    
    # Save results with permutation ID
    participant_id = f"avg_perm{permutation_id:03d}"
    save_results(
        clinical_group_name=clinical_group,
        participant=participant_id,
        seed=model.seed_region,
        alpha=None,  # NDM doesn't use alpha parameter
        r=r,
        SSE=SSE,
        model_output=model_output,
        prediction=prediction,
        optimization_iters=df,  # Store the seed optimization dataframe as iters
        model="NDM_REWIRED"
    )
    
    return dict(
        permutation_id=permutation_id,
        optimal_parameters={"seed": model.seed_region},
        SSE=SSE,
        r=r,
        corr_metrics=corr_metrics  # Will be None if using existing file
    )


def run_rewiring_analysis(clinical_group="CN", n_permutations=100, n_jobs=-1, 
                         bin_swaps=5, wei_freq=0.1):
    """
    Run rewiring analysis on connectomes using NDM.
    
    Args:
        clinical_group: Clinical group to analyze (default: CN - best performing group)
        n_permutations: Number of permutations to run
        n_jobs: Number of parallel jobs
        bin_swaps: Number of binary swaps for rewiring
        wei_freq: Frequency of weight sorting in rewiring
    """
    # Run tasks sequentially using a for loop
    results = []
    for perm_id in tqdm(range(1, n_permutations + 1), desc="Running NDM with rewired connectomes"):
        result = run_ndm_rewired(
            clinical_group, 
            perm_id, 
            bin_swaps=bin_swaps,
            wei_freq=wei_freq
        )
        results.append(result)
    
    # Compile results into a DataFrame
    results_df = pd.DataFrame(results)
    
    # Save summary results
    summary_path = os.path.join(constants.results_folder, f"ndm_rewired_{clinical_group}_summary.csv")
    results_df.to_csv(summary_path, index=False)
    print(f"Results summary saved to {summary_path}")
    
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NDM model with rewired connectomes")
    parser.add_argument("--clinical_group", type=str, default="CN", help="Clinical group to analyze (default: CN)")
    parser.add_argument("--n_permutations", type=int, default=100, help="Number of permutations")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs (-1 for all cores)")
    parser.add_argument("--bin_swaps", type=int, default=10, help="Number of binary swaps for rewiring")
    parser.add_argument("--wei_freq", type=float, default=0.1, help="Frequency of weight sorting in rewiring")
    
    args = parser.parse_args()
    
    # Run the rewiring analysis
    results = run_rewiring_analysis(
        clinical_group=args.clinical_group,
        n_permutations=args.n_permutations,
        n_jobs=args.n_jobs,
        bin_swaps=args.bin_swaps,
        wei_freq=args.wei_freq
    )