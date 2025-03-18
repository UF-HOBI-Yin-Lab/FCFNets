import torch as t

# Set the parameter variables and change them uniformly here
class config:
    def __init__(self):
        self.encoder_hid_dim=64#128#64#32#16
        self.encoder_out_dim=32#64#32#16#8 
        self.pred_hid_dim=16#32#4#16#8#4
        self.pred_out_dim=1 
        self.decoder_hid_dim=32#64#8#32#16#8
        self.upsample_type='hybrid'#default, random, similar, dissimilar, cluster, hybrid
        self.mixed_ratio='8-2'
        self.pos_neg_ratio=0.9
        self.pred_thres=0.5
        self.pred_cf_thres=0.5
        self.drop=0.3#0.3
        #model
        self.model_name = 'VCNet'
        # Set random seeds
        self.seed = 42# 42 6657 2024 123 666
        self.fold = 1# 1 2 3 4 5
        # Training parameters
        self.batchSize = 16
        self.num_epochs = 80
        self.lr = 0.001
        self.weight_decay = 0.005#0.001
        self.earlyStop = 20
        self.explain_mode = 'shap'
        self.datapath = 'dataset/MPData/tem/hf_manual_misrate_data_strategy_strfinal.csv' # final, v5lbl 
        self.device = t.device("cuda:0")
        if self.upsample_type == 'hybrid':
            self.savePath=f"checkpoints/models/model_{self.model_name}_seed{self.seed}_fold{self.fold}/data{self.datapath[self.datapath.rindex('_')+1:self.datapath.index('.')]}_enh{self.encoder_hid_dim}_eno{self.encoder_out_dim}_pdh{self.pred_hid_dim}_pdo{self.pred_out_dim}_dch{self.decoder_hid_dim}_b{self.batchSize}_d{self.drop}_thres{self.pred_thres}_cfthres{self.pred_cf_thres}_{self.upsample_type}{self.mixed_ratio}_pnr{self.pos_neg_ratio}/" 
        else:
            self.savePath=f"checkpoints/models/model_{self.model_name}_seed{self.seed}_fold{self.fold}/data{self.datapath[self.datapath.rindex('_')+1:self.datapath.index('.')]}_enh{self.encoder_hid_dim}_eno{self.encoder_out_dim}_pdh{self.pred_hid_dim}_pdo{self.pred_out_dim}_dch{self.decoder_hid_dim}_b{self.batchSize}_d{self.drop}_thres{self.pred_thres}_cfthres{self.pred_cf_thres}_{self.upsample_type}_pnr{self.pos_neg_ratio}/" 