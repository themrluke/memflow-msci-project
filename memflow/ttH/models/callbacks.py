# cfm_sampling_callback.py
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader
from models.utils import compare_distributions

# models/callbacks.py

from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from .utils import compare_distributions, plot_sampling_distributions


class CFMSamplingCallback(Callback):
    def __init__(self, dataset, freq=5, steps=10, show_sampling_distributions=False):
        """
        dataset: e.g. combined_dataset_valid
        freq   : how often (epochs) to do sampling
        steps  : bridging steps for Euler
        """
        super().__init__()
        self.dataset = dataset
        self.freq = freq
        self.steps = steps
        self.show_sampling_distributions = show_sampling_distributions

    def move_batch_to_device(batch, device):
        new_batch = {}
        for top_key, top_val in batch.items():
            new_top_dict = {}
            for key, val in top_val.items():
                if isinstance(val, list):
                    new_top_dict[key] = [item.to(device) for item in val]
                elif isinstance(val, torch.Tensor):
                    new_top_dict[key] = val.to(device)
                else:
                    new_top_dict[key] = val
            new_batch[top_key] = new_top_dict
        return new_batch

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.freq == 0:
            loader = DataLoader(self.dataset, batch_size=32, shuffle=False)
            batch = next(iter(loader))
            batch = self.move_batch_to_device(batch, pl_module.device)

            with torch.no_grad():
                 # Sample multiple times for distribution plots
                num_samples = 100  # Define how many times to sample for the distribution plot
                gen_data_samples = [pl_module.sample(batch, steps=self.steps) for _ in range(num_samples)]

                # Single-time sampling for comparison plots
                gen_data_list = pl_module.sample(batch, steps=self.steps)

             # Compare type=0 (partons->jets)
            real_data = batch["reco"]["data"][0]  # (B, n_recoJets, feats)
            gen_data = gen_data_list[0]          # (B, n_hardPartons, feats)

            # Call `compare_distributions`
            compare_distributions(real_data, gen_data, feat_idx=0, feat_name="pt")
            compare_distributions(real_data, gen_data, feat_idx=1, feat_name="eta")
            compare_distributions(real_data, gen_data, feat_idx=2, feat_name="phi")

            # Call `plot_sampling_distributions` if enabled
            if self.show_sampling_distributions:
                plot_sampling_distributions(
                    real_data=real_data,
                    gen_data_samples=[sample[0] for sample in gen_data_samples],  # Only type 0 (jets)
                    feat_names=["pt", "eta", "phi"],
                    event_idx=0  # Choose which event to plot
                )

            # Potentially log figure to Comet, etc.
            # trainer.logger.experiment.log_figure(...)