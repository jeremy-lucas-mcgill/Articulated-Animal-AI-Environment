from stable_baselines3.common.callbacks import BaseCallback
import torch

class TerminateOnThresholdNanInfCallback(BaseCallback):
    def __init__(self, nan_threshold=0.05, inf_threshold=0.05, verbose=0):
        """
        Stops training if the fraction of NaN or Inf values in model parameters
        exceeds the specified thresholds.
        
        Args:
            nan_threshold (float): Fraction of NaN values (between 0 and 1) allowed before termination.
            inf_threshold (float): Fraction of Inf values (between 0 and 1) allowed before termination.
            verbose (int): Verbosity level.
        """
        super(TerminateOnThresholdNanInfCallback, self).__init__(verbose)
        self.nan_threshold = nan_threshold
        self.inf_threshold = inf_threshold

    def _on_step(self) -> bool:
        total_params = 0
        total_nan = 0
        total_inf = 0

        for param in self.model.parameters():
            param_data = param.detach()
            num_elements = param_data.numel()
            total_params += num_elements
            total_nan += torch.isnan(param_data).sum().item()
            total_inf += torch.isinf(param_data).sum().item()

        if total_params == 0:
            return True  # Avoid division by zero

        nan_ratio = total_nan / total_params
        inf_ratio = total_inf / total_params

        if nan_ratio > self.nan_threshold or inf_ratio > self.inf_threshold:
            print(
                f"Terminating training: {nan_ratio*100:.2f}% NaN and "
                f"{inf_ratio*100:.2f}% Inf in model weights."
            )
            return False  # Stop training

        return True  # Continue training

class TerminateOnSingleNanInfCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TerminateOnSingleNanInfCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Iterate through all parameters of the model
        for param in self.model.parameters():
            # Check if any parameter has a NaN or Inf value
            if torch.any(torch.isnan(param)) or torch.any(torch.isinf(param)):
                print("NaN or Inf detected in model weights. Terminating training!")
                return False  # This signals the training loop to stop
        return True