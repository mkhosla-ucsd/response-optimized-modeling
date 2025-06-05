import os
import torch
import torch.nn.functional as F
from models_brain import C8NonSteerableCNN, C8SteerableCNN
from neuralpredictors.layers.readouts import SpatialXFeatureLinear, FullGaussian2d
from scipy.stats import pearsonr
from utils import save_checkpoint
from train_utils import *
from model_utils import *
from dataloaders import Dataset_visual
import argparse
from readouts import SemanticSpatialTransformer, GlobalLinear, AttentionLinear
from itertools import repeat
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encoding model')
    parser.add_argument('--roi_type', default='face', type=str, help='Roi: face, place, words, bodies')
    parser.add_argument('--roi', default=1, type=int, help='Roi number')
    parser.add_argument('--feats', default=48, type=int, help='Number of features')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--readout_type', default='factorized', type=str, help='Readout type')
    parser.add_argument('--is_eq', default = "eq", type = str, help = 'Equivariance structure')
    parser.add_argument('--gpu', default = "0", type = str, help = 'GPU device')
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    roi_type = args.roi_type
    roi = args.roi
    method = "scratch"
    readout_type = args.readout_type 
    
    
    model_dir = './saved_models'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    model_base = 'ho_%d_%s_%d_%s_%s' % (args.feats, roi_type, roi, method, readout_type)
    
    params = {'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 6}
   
    
    if args.is_eq == 'eq':
        print('Training equivariant model')
        core = C8SteerableCNN(n_feats = args.feats) 
    elif args.is_eq == 'noneq':
        print('Training non-equivariant model')   
        core = C8NonSteerableCNN(n_feats = args.feats)
    else:
        print("Invalid is_eq value")
        raise
        
    # Generators
    training_set = Dataset_visual(mode = 'train', roi_type = roi_type, roi = roi)
    training_generator = torch.utils.data.DataLoader(training_set,  **params)

    validation_set = Dataset_visual(mode = 'val', roi_type = roi_type, roi = roi)
    validation_generator = torch.utils.data.DataLoader(validation_set,  **params)
    
   
    n_neurons = training_set.responses.shape[1]
    print('Number of neurons', n_neurons)
    

    if readout_type =='factorized':
        readout = SpatialXFeatureLinear(core(torch.randn(1, 3, 224, 224)).size()[1:], n_neurons,  bias = True)  
    elif readout_type =='attention':
        readout = AttentionLinear(core(torch.randn(1, 3, 224, 224)).size()[1:], n_neurons,  bias = True)
    elif readout_type =='global':
        readout = GlobalLinear(core(torch.randn(1, 3, 224, 224)).size()[1:], n_neurons,  bias = True)
    else:
        print('Invalid readout type | Not implemented')
        raise 
        
    
    model = Encoder(core, readout)
    model.cuda()
    
    
    ################ Define training parameters ################
    schedule = [1e-4]
    criterion = masked_MSEloss 
    best_corr = 0
    patience = 20
    iter_tracker = 0 
    accumulate_gradient= 4
    n_epochs = 100
    
    ############## Start training ##########
    for opt, lr in zip(repeat(torch.optim.Adam), schedule):
        print('Training with learning rate', lr)
        optimizer = opt(model.parameters(), lr=lr)
        optimizer.zero_grad()
        iteration = 0
        restore_file = 'best_' + model_base
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')


            
        assert accumulate_gradient > 0, 'accumulate_gradient needs to be > 0'
        for epoch in range(n_epochs):
            for x_batch, y_batch in training_generator:
         
                obj = full_objective(model, x_batch.cuda().float(), y_batch.cuda().float(), criterion)
                obj.backward()
          
                iteration += 1 
                
                if iteration % accumulate_gradient == accumulate_gradient - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                if iteration % 100 == 0:
                    model.eval()
                    true, preds = compute_predictions(validation_generator, model)

                    val_corr = compute_scores(true, preds)
                    print('Val correlation:', val_corr)
                    model.train()
                    is_best = val_corr >= best_corr
                    if is_best:
                        best_corr = val_corr.copy()
                        iter_tracker = 0        
                        save_checkpoint({'epoch': epoch + 1,
                                               'state_dict': model.state_dict(),
                                               'optim_dict' : optimizer.state_dict()},
                                               is_best = is_best,
                                               checkpoint = model_dir, model_str = model_base)
                    else:
                        iter_tracker += 1
                        if iter_tracker == patience: 
                            print('Training complete')
                            break 
            if iter_tracker == patience: 
                            print('Training complete')
                            break  



