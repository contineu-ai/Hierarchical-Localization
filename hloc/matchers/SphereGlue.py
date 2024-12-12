import sys
from pathlib import Path
import torch
from ..utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / "../../third_party"))
from SphereGlue.model.sphereglue import SphereGlue


class SG(BaseModel):
    def _init(self,conf):
        default_config = {'K': 2, #Chebyshev filter size
                    'GNN_layers': ['cross'], 
                    'match_threshold': 1,
                    'sinkhorn_iterations':20,
                    'aggr': 'add',
                    'knn': 20,
                    'max_kpts': 20000
                }
        default_config['descriptor_dim'] = 256
        default_config['output_dim'] = 256*2
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        matching_test = SphereGlue(default_config).to(device)              
        model_path = '/root/Hierarchical-Localization/third_party/SphereGlue/model_weights/' + 'superpoint' + '/autosaved.pt'
        print (model_path)
        ckpt_data = torch.load(model_path)
        matching_test.load_state_dict(ckpt_data["MODEL_STATE_DICT"])
        matching_test.eval()
        self.net = matching_test
    def _forward(self, data):
        return self.net(data)
