
This repo contains code for training response optimized models and dissecting their hidden unit activations 

A note on dataloaders: 

Dataloaders expect the following data: 

stimuli_all_subs.npy : A numpy array of dimensions (37,000 x 224 x 224 x 3) containing all 37,000 images for subjects in NSD with IDs 1,2,5,7: These subjects had full 3 repetitions for 10,000 images 


response_all_subs_roi_[roi_type]_val[roi_num].npy: A numpy array of dimensions 37,000 x num_voxels where num_voxels indicate the number of voxels in each ROI (roi_type indicates the type of roi, i.e. whether it is selective to faces, places, words or bodies. And roi_num indicates the ROI number used to localize the ROI in the floc experiment. 
FFA (FFA1-2) has a label of [2,3] in floc for faces
RSC has a label of [3] in floc for places
VWFA has a label of [2] in floc for words
EBA has a label of [1] in floc for bodies

Key dependencies: 
e2cnn: https://github.com/QUVA-Lab/e2cnn
neuralpredictors: https://github.com/sinzlab/neuralpredictors
network dissection: https://github.com/davidbau/dissect

## Demo data 
The data directory contains some demo data to play around with. It contains stimuli and corresponding FFA response (across all 4 subjects) for a subset of 100 stimuli from the test set. 
The python notebook 'Demo training model.ipynb' provides demo code for training response optimized models 

#### Output after training models 
All trained models will be saved in the saved_models directory by default

### ROI number indices the number assigned to the ROI in the functional localizer masks
# FFA
python train_response_optimized.py --roi_type face --roi 4
# VWFA 
python train_response_optimized.py --roi_type word --roi 2
# RSC
python train_response_optimized.py --roi_type place --roi 3
# EBA
python train_response_optimized.py --roi_type bodies --roi 1



#### Code for dissecting models 
The python notebook 'Demo dissection visualization.ipynb' provides demo code illustrating network dissection
# FFA
python dissect_encoding_model.py --roi_type face --roi 4
# VWFA
python dissect_encoding_model.py --roi_type word --roi 2
# RSC
python dissect_encoding_model.py --roi_type place --roi 3
# EBA
python dissect_encoding_model.py --roi_type bodies --roi 1

#### Output after network dissection
This will create 3 files for each model condi99.npz, topk.npz, rq.npz which store the quantile thresholds and the the IOU scores for each of the concepts in the network dissection dictionary. 
The matchingg concept names are stored in label_list.npy. 
 
