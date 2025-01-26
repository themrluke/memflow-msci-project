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
from .utils import compare_distributions

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


class CFMSamplingCallback(Callback):
    def __init__(self, dataset, freq=5, steps=10):
        """
        dataset: e.g. combined_dataset_valid
        freq   : how often (epochs) to do sampling
        steps  : bridging steps for Euler
        """
        super().__init__()
        self.dataset = dataset
        self.freq    = freq
        self.steps   = steps

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.freq == 0:
            loader = DataLoader(self.dataset, batch_size=32, shuffle=False)
            batch = next(iter(loader))
            batch = move_batch_to_device(batch, pl_module.device)

            with torch.no_grad():
                gen_data_list = pl_module.sample(batch, steps=self.steps)

            # Suppose we compare type=0 (partons->jets)
            real_data = batch["reco"]["data"][0]  # (B, n_recoJets, feats)
            gen_data  = gen_data_list[0]         # (B, n_hardPartons, feats)

            # Check shapes or disclaim if they differ
            # We'll do a single histogram for demonstration:
            compare_distributions(real_data, gen_data, feat_idx=0, feat_name="pt")
            # Potentially log figure to Comet, etc.
            # trainer.logger.experiment.log_figure(...)