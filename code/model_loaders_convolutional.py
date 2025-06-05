import sys
from models_brain import C8NonSteerableCNN, C8SteerableCNN
from neuralpredictors.layers.readouts import SpatialXFeatureLinear
from dataloaders import Dataset_visual
import os
import torch
import torch.nn.functional as F
from train_utils import *
from model_utils import *
from scipy.stats import pearsonr
from itertools import repeat


params = {'batch_size': 4,
          'shuffle': True,
          'num_workers': 6}
    
    
class ConvolutionalNeuron(nn.Module):
    
    def __init__(self, model, key=None):
        super().__init__()
        model = model.cuda() 
        model.eval()
        self.key = key
        self.core = model.core
        self.features = model.readout.features.transpose(0,1)
        self.bias = model.readout.bias.view(1, -1, 1, 1) 
        print(self.bias.shape)
    def forward(self, x):
        x = self.core(x)
        x = torch.einsum('bcwh,cn->bnwh', x, self.features) + self.bias
        return x
        
class NeuronLayer(nn.Module):
    
    def __init__(self, model, key = None):
        super().__init__()
        model = model.cuda() 
        model.eval()
        self.key = key
        self.features = model.readout.features.transpose(0,1)
        self.bias = model.readout.bias.view(1, -1, 1, 1) 
        print(self.bias.shape)
    def forward(self, x):
        x = torch.einsum('bcwh,cn->bnwh', x, self.features[:, self.key]) + self.bias[:, self.key]
        return x
    
class Encoder_neuron(nn.Module):
    def __init__(self, core, neuronlayer):
        super().__init__()
        self.core = core
        self.neuronlayer = neuronlayer
       

    def forward(self, x, **kwargs):
        x = self.core(x)
        x = self.neuronlayer(x)
        return x     
    


##########
def loadmodel_convolutional_response_optimized(roi_type = 'face', roi = 2, method = 'scratch',  readout_type = 'factorized', feats = 48, key = None, thr = 0.1):
    params = {'batch_size': 4,
          'shuffle': False,
          'num_workers': 6}
    test_set = Dataset_visual(mode = 'test', roi_type = roi_type, roi = roi)
    test_generator = torch.utils.data.DataLoader(test_set,  **params)
    n_neurons = test_set.responses.shape[1]
    
    model_dir = './saved_models/'
    model_base = 'ho_%d_%s_%d_%s_%s' % (feats, roi_type, roi, method, readout_type)
    
    core = C8SteerableCNN(n_feats = feats)
    core.cpu()
    readout = SpatialXFeatureLinear(core(torch.randn(1, 3, 224, 224)).size()[1:], n_neurons,  bias = True)
    predictor = Encoder(core, readout)
    

    restore_file = 'best_' + model_base
    restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
    checkpoint = torch.load(restore_path)
    state_dict = checkpoint['state_dict']
    predictor.load_state_dict(state_dict, strict=False)
    predictor.cuda()
    predictor.eval()
    
    
    true, preds = compute_predictions(test_generator, predictor)
    
    test_corr = np.asarray([pearsonr(true[:,i], preds[:,i])[0] for i in range(true.shape[1])]) 
    print('Test correlation:', np.nanmean(test_corr)) 
    if thr is not None:
        key = np.where(test_corr>thr)[0]
    else: 
        key = np.arange(len(test_corr))
    neuron_layer = NeuronLayer(predictor, key)  
    
    #### Convert model to be fully convolutional 
    model = Encoder_neuron(predictor.core, neuron_layer)

    model.cuda()        
    model.eval()
    return model, test_corr, true, preds



