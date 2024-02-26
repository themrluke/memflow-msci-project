import torch.nn as nn
import torch.nn.functional as F


class Classifier_nojets(nn.Module):
    def __init__(self, hidden_features, dim_feedforward_transformer,
                nhead_encoder, no_layers_encoder, dtype):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_features,
                                                   dim_feedforward= dim_feedforward_transformer,
                                                   nhead=nhead_encoder,
                                                   batch_first=True,
                                                   dtype=dtype)

        self.transformer_encoder = nn.TransformerEncoder(
                                        encoder_layer, num_layers=no_layers_encoder)
        
        self.firstproj = nn.Linear(in_features=3,
                            out_features=1, # (pt//eta/phi) -> 1
                            dtype=dtype)
        
        self.proj = nn.Linear(in_features=4,
                            out_features=15, # no objs: 4 ... 18 (jets + met + lepton)
                            dtype=dtype)
        

        
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, input_transformer):
        transformer_output = self.transformer_encoder(input_transformer)
        project_each_parton = self.firstproj(self.gelu(transformer_output))
        
        decoder_output = self.softmax(self.proj(self.gelu(project_each_parton[...,0])))
            
        return decoder_output