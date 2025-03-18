import torch
import torch.nn as nn

class VCNet(nn.Module):
    def __init__(self, input_dim, encoder_hid_dim, encoder_out_dim, pred_hid_dim, pred_out_dim, decoder_hid_dim, drop=0.3):
        super(VCNet, self).__init__()
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_hid_dim),
            # nn.BatchNorm1d(encoder_hid_dim),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(encoder_hid_dim, encoder_out_dim),
            # nn.BatchNorm1d(encoder_out_dim),
            nn.ReLU()
        )
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(encoder_out_dim, pred_hid_dim),
            nn.BatchNorm1d(pred_hid_dim),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(pred_hid_dim, pred_out_dim),
            nn.Sigmoid()
        )
        # Decoder for counterfactual generation
        self.decoder = nn.Sequential(
            nn.Linear(decoder_hid_dim+pred_out_dim, decoder_hid_dim), # Adjusted input size to match concatenated z_cf size (64 + 1 = 65)
            # nn.BatchNorm1d(decoder_hid_dim),
            nn.ReLU(),
            nn.Linear(decoder_hid_dim, input_dim)
        )

    def forward(self, x, target_class=None):
        z = self.encoder(x)  # Encoded features
        pred = self.predictor(z)  # Predictions

        if target_class is not None:
            # Expand target_class to match z's dimensions
            target_class_expanded = target_class.view(-1, 1).expand(-1, 1)
            z_cf = torch.cat([z, target_class_expanded], dim=1)  # Concatenate along feature dimension
            x_cf = self.decoder(z_cf)  # Counterfactual samples
            return pred, x_cf

        return pred, None
    
