This repo contains code for training response optimized models and dissecting their hidden unit activations. 
Methodological details and results are described in this paper: https://www.biorxiv.org/content/10.1101/2022.03.16.484578v1

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
Demo data can be accessed from the following google drive link: https://drive.google.com/drive/folders/1Qu2pmoF3JguZJbDjOiHI9v_njAs1HERo?usp=sharing 
You can download and put this data in the data directory to play around with the code. It contains stimuli and corresponding FFA response (across all 4 subjects) for a subset of 100 stimuli from the test set. 
The python notebook 'Demo training model.ipynb' provides demo code for training response optimized models. 
Demo data

## Full data 
Full preprocessed brain response data for the ROIs modeled in this study can be accessed using the following link: https://drive.google.com/file/d/1FcY252RMhdh0E4U7BblfLsk74v74rkUZ/view?usp=sharing 
The directory also contains the COCO IDs corresponding to these brain responses. They can be downloaded from [MS COCO Home Page](https://cocodataset.org/#home) or via NSD. 

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
 
