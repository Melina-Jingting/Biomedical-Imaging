from bct import clustering_coef_wu, efficiency_wei, strengths_und, diffusion_efficiency, density_und
import pandas as pd 
import os
import seaborn as sns

# Get directory of the constants.py file
current_file_dir = os.path.dirname(os.path.abspath(__file__))

clinical_group_names = ["CN","EMCI","LMCI","DEM"]
clinical_group_colors = sns.color_palette("RdYlGn_r", 4).as_hex()
clinical_group_colors_map = dict(zip(clinical_group_names, clinical_group_colors))
model_colors_map = {
    "NDM": "#a6cee3",     # light blue
    "FKPP": "#1f78b4",     # medium blue
    "WFKPP": "#08306b"     # dark blue; mapped to A$\beta$-FKPP in model_name_map
}
metric_functions = dict(
    node_strength = strengths_und,
    clustering_coefficient = clustering_coef_wu,
    density = lambda x: density_und(x)[0],
    global_efficiency = efficiency_wei,
    diffusion_efficiency = lambda x: diffusion_efficiency(x)[0], #extracting only the mean
)
global_metrics = ["global_efficiency", "density", "diffusion_efficiency"]
node_level_metrics = ["node_strength", "clustering_coefficient"]

# file paths
root_folder = os.path.dirname(current_file_dir) + "/"
figures_folder = root_folder + "writeup/figures/"
optimal_parameters_filepath = root_folder + "results/optimal_parameters/{model}.json"

model_output_filepath = root_folder + "results/model_output/{model}/{model}_{clinical_group_name}_{participant}.npy"
prediction_filepath = root_folder + "results/prediction/{model}/{model}_{clinical_group_name}_{participant}.npy"
optimization_iters_filepath = root_folder + "results/optimization_iters/{model}/{model}_{clinical_group_name}_{participant}.csv"
residuals_filepath = root_folder + "results/residuals/{model}_{clinical_group_name}_{participant}.npy"

connectome_filepath = root_folder + "data/connectomes/{clinical_group_name}/{clinical_group_name}_{participant}.csv"
average_connectome_path = root_folder + "data/connectomes/{clinical_group_name}/{clinical_group_name}_avg.csv"
tau_filepath = root_folder + "data/PET/tau_ab+_tau+.csv"
amyloid_filepath = root_folder + "data/PET/amyloid_ab+_tau+.csv"

tau_region_fs_label = pd.read_csv(root_folder + "data/TauRegionList.csv")["FS_label"].tolist()
tau_region_raj_label = pd.read_csv(root_folder + "data/TauRegionList.csv")["Raj_label"].tolist()

#FKPP seed_list
fkpp_seed_list = ['Inferiortemporal', 'Middletemporal', 'Temporalpole', 'Amygdala', 'Entorhinal']
wfkpp_seed_list = ['Inferiortemporal', 'Entorhinal', 'Amygdala', 'Temporalpole']
fkpp_rewired_seed_list = ['Inferiortemporal', 'Fusiform', 'Middletemporal', 'Bankssts',
       'Entorhinal', 'Inferiorparietal', 'Lateraloccipital',
       'Cerebellum_Cortex', 'Thalamus_Proper', 'Caudate', 'Hippocampus',
       'Parahippocampal', 'Medialorbitofrontal', 'Putamen',
       'Rostralmiddlefrontal', 'Cuneus', 'Lateralorbitofrontal',
       'Supramarginal', 'Isthmuscingulate', 'Caudalmiddlefrontal']

#Mapping
model_name_map = {
    "NDM": "NDM",
    "FKPP": "FKPP",
    "WFKPP": r"A$\beta$-FKPP",
    "MFKPP-WFKPP": r"FKPP(A$\beta$-FKPP parameters) - A$\beta$-FKPP",
}
clinical_group_name_map = {
    "CN":"CN",
    "EMCI":"EMCI",
    "LMCI":"LMCI",
    "DEM":"AD"
}