import os 
import numpy as np 
import pandas as pd
import json 
from my_src import constants

def load_connectome(clinical_group, participant, folder_path=constants.connectome_filepath):
    connectome_filepath = constants.connectome_filepath.format(clinical_group_name=clinical_group, participant=participant)
    connectome = np.loadtxt(connectome_filepath, delimiter=',')
    return connectome

def normalise_data(data):
    ''' min-max normalise the data '''
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def save_results(clinical_group_name,
                 participant,
                 seed,
                 alpha,
                 r,
                 SSE,
                 model_output,
                 prediction,
                 optimization_iters,
                 model,
                 optimal_parameters_filepath = constants.optimal_parameters_filepath,
                 model_output_filepath = constants.model_output_filepath,
                 prediction_filepath = constants.prediction_filepath,
                 optimization_iters_filepath = constants.optimization_iters_filepath,):
    
    # Prepare entry to save
    entry = dict(
        clinical_group_name = clinical_group_name,
        participant = participant,
        seed_region = seed,
        alpha = alpha,
        r = r,
        SSE = SSE
    )
    
    # Format filepaths with the provided keys
    optimal_parameters_filepath = optimal_parameters_filepath.format(model=model)
    model_out_fp = model_output_filepath.format(model=model, clinical_group_name=clinical_group_name, participant=participant)
    prediction_fp = prediction_filepath.format(model=model, clinical_group_name=clinical_group_name, participant=participant)
    optimization_iters_fp = optimization_iters_filepath.format(model=model, clinical_group_name=clinical_group_name, participant=participant)
    
    # Load existing JSON file (if it exists)
    if os.path.exists(optimal_parameters_filepath):
        with open(optimal_parameters_filepath, "r") as f:
            entries = json.load(f)
    else:
        entries = []
    
    # Check if an entry with the same clinical_group_name and participant exists.
    updated = False
    new_entries = []
    for e in entries:
        if e.get("clinical_group_name") == clinical_group_name and e.get("participant") == participant:
            new_entries.append(entry)  # update the entry
            updated = True
        else:
            new_entries.append(e)
    if not updated:
        new_entries.append(entry)
    
    # Write the updated entries back as valid JSON
    with open(optimal_parameters_filepath, "w") as f:
        json.dump(new_entries, f, indent=2)
    
    # Save model output
    np.save(model_out_fp, model_output)
    
    # Save prediction
    np.save(prediction_fp, prediction)
    
    # Save optimization iterations
    if "NDM" in model:
        optimization_iters.to_csv(optimization_iters_fp, index=False)
    else:
        x_iters = pd.DataFrame(optimization_iters["x_iters"], columns=["node","alpha"])
        x_iters["func_vals"] = optimization_iters["func_vals"]
        x_iters.to_csv(optimization_iters_fp, index=False)


def ggseg_plot_and_save(residuals, filepath):
    CORT_IDX = np.concatenate([np.arange(34), np.arange(49, 83)])
    region_list = pd.read_csv( "../data/TauRegionList_ggseg.csv")["Raj_label_ggseg"].tolist()  
    ## naming convention for ggseg (https://github.com/ggseg/python-ggseg/tree/main/ggseg/data/dk)
    ctx_region_list = list(map(lambda x: region_list[x],CORT_IDX))
    res_d = dict(zip(region_list, residuals))
    
def load_optimal_parameters_df(model_names=constants.model_name_map.keys()):
    optimal_parameters_dfs = []
    for model_name in model_names:
        with open(constants.optimal_parameters_filepath.format(model=model_name), "r") as f:
            optimal_parameters_df = pd.DataFrame(json.load(f))
            optimal_parameters_df["model_name"] = model_name
            optimal_parameters_dfs.append(optimal_parameters_df)
    optimal_parameters_df = pd.concat(optimal_parameters_dfs)
    return optimal_parameters_df

def get_optimal_parameters(optimal_parameters_df, model_name, participant, clinical_group_name):
    optimal_parameters_row = optimal_parameters_df[(optimal_parameters_df["model_name"] == model_name) \
    & (optimal_parameters_df["participant"]==participant)\
        & (optimal_parameters_df["clinical_group_name"]==clinical_group_name)].iloc[0]
    return optimal_parameters_row