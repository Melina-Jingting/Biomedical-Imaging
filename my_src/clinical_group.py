import abc 
import numpy as np
from bct import normalize 
from my_src import constants 
from my_src import utils
import my_src.utils, my_src.constants
from joblib import Parallel, delayed


class ClinicalGroup(abc.ABC):
    def __init__(
            self,
            name: str,
            n_participants: int = 42,
    ):
        self.name = name 
        self.n_participants = n_participants
        self.connectomes = self._load_normalized_connectomes()
        self.average_connectomes = self._compute_average_connectomes()
        self._compute_metrics(constants.metric_functions.keys())
        self._compute_average_and_separate_nodewise_metrics()
    
    def _compute_metric_single_patient(self, metric_name, i):
        try:
            metric_function = constants.metric_functions[metric_name]
            connectome = self.connectomes[i].copy()
            
            # Try to compute the metric
            result = metric_function(connectome)
            return result
            
        except Exception as e:
            # Log the error
            print(f"Error computing {metric_name} for participant {self.name}:{i} - {str(e)}")
            
            # Create appropriate NaN placeholder based on expected output shape
            # First check if we've successfully computed this metric for another participant
            for j in range(self.n_participants):
                if j != i:
                    try:
                        sample_result = metric_function(self.connectomes[j].copy())
                        # Create NaN array with the same shape
                        return np.full_like(sample_result, np.nan, dtype=float)
                    except:
                        continue
            
            # If we couldn't determine shape from other participants, use these defaults
            if "node" in metric_name.lower():  # Likely node-wise metric
                return np.full(self.connectomes[i].shape[0], np.nan)
            else:  # Likely scalar metric
                return np.nan
            
    def _compute_metrics(self, metric_names: list):
        self.metrics = dict()
        for metric_name in metric_names:
            metrics = Parallel(n_jobs=-1)(
                delayed(self._compute_metric_single_patient)(metric_name, i)
                for i in range(self.n_participants)
            )
            self.metrics[metric_name] = np.array([m for m in metrics if m is not None])
    
    def _compute_average_and_separate_nodewise_metrics(self):
        nodewise_metrics = []
        for metric_name in self.metrics.keys():
            if self.metrics[metric_name].ndim == 2:
                nodewise_metrics.append(metric_name)
        
        for metric_name in nodewise_metrics:
            self.metrics[f"{metric_name} (avg)"] = np.mean(self.metrics[metric_name], axis = 0)
            for i, node_name in enumerate(constants.tau_region_raj_label):
                self.metrics[f"{metric_name} ({node_name})"] = self.metrics[metric_name][:,i]
            self.metrics.pop(metric_name)
                
    
    def _load_normalized_connectomes(self):
        return np.array(
            [normalize(
                utils.load_connectome(self.name,i)
                ) 
            for i in range(self.n_participants)]
        )

    def _compute_average_connectomes(self):
        return np.mean(self.connectomes, axis = 0)
    