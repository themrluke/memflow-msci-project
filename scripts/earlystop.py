import torch
from comet_ml.integration.pytorch import log_model

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = torch.inf

    def early_stop(self, validation_loss, model_weights, optimizer_weights, modelName, exp_log):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0

            model_checkpoint = {
                'model_state_dict': model_weights,
                'optimizer_state_dict': optimizer_weights,
                'val_loss': validation_loss
            }
            
            torch.save(model_checkpoint, modelName)
            
            log_model(exp_log, model_checkpoint, model_name=modelName)

        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True

        return False
