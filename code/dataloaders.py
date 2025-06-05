import numpy as np
import torch
import os 
import pickle 
from skimage.transform import resize
from torchvision import transforms
from PIL import Image


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



class Dataset_visual(torch.utils.data.Dataset):

    def __init__(self, mode = 'train', roi_type = None, roi = None, dim=(224, 224),  n_channels = 3, 
                data_path = '../data/'):
           
        splits = np.load(os.path.join(data_path,'splits_all.npy'), allow_pickle = True).item()
        self.ids = splits[mode]
        self.responses = np.load(os.path.join(data_path,'response_all_subs_roi_%s_val%d.npy'%(roi_type, roi)), mmap_mode = 'r')
        self.images = np.load(os.path.join(data_path,'stimuli_all_subs.npy'), mmap_mode = 'r')   
        self.dim = dim
        self.total_size = len(self.ids) 
        self.n_channels = n_channels
        self.n_neurons =  self.responses.shape[1]


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ids) 
  
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        data_ix = self.ids[index]
        X = np.asarray(self.images[data_ix]).astype('float32') 
        y = np.asarray(self.responses[data_ix])
        X = preprocess(X)
        return X, y 



    
