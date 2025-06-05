import os 
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from importlib import reload
import IPython
import settings
import torch, argparse, os, shutil, inspect, json, numpy, math
import netdissect
import argparse
import numpy as np
import signal
import time

from model_loaders_convolutional import loadmodel_convolutional_response_optimized
from netdissect.easydict import EasyDict
from netdissect import pbar, nethook, renormalize, parallelfolder, pidfile
from netdissect import upsample, tally, imgviz, imgsave, bargraph, show
from experiment import dissect_experiment as experiment
from feature_operation_original import hook_feature 
from netdissect import pbar, nethook
from urllib.request import urlopen
from netdissect import renormalize
from netdissect import renormalize
from netdissect import imgviz
mpl.rcParams['lines.linewidth'] = 0.25
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.linewidth'] = 0.25



args = EasyDict(model='steerablecnn', dataset='fullNSD', seg='netpq', layer='voxel', quantile=0.01) 

class TimeOutException(Exception):
    pass

def alarm_handler(signum, frame):
    print("ALARM signal received")
    raise TimeOutException()

def loop(n):
    for sec in range(n):
        print("sec {}".format(sec))
        time.sleep(1)
 


def inc_forever():
    print('Starting function inc_forever()...')
    while True:
        time.sleep(1)
        print(next(counter))
        
def return_seglabels():
    print('Starting function return_seglabels()...')
    segmodel, seglabels, segcatlabels = experiment.setting.load_segmenter(args.seg)
    return segmodel, seglabels, segcatlabels 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dissecting model')
    parser.add_argument('--roi_type', default='face', type=str, help='Roi: face, place, words, bodies')
    parser.add_argument('--roi', default=-2, type=int, help='Roi number')
    arg = parser.parse_args()

    
    
    
    resdir = 'results/%s_%d' % (arg.roi_type, arg.roi)
    def resfile(f):
        return os.path.join(resdir, f)
    
    
    model = loadmodel_convolutional_response_optimized(roi_type = arg.roi_type, roi = arg.roi)
 
    model = nethook.InstrumentedModel(model[0]).cuda().eval()
    model.retain_layer('neuronlayer')
    layername = 'neuronlayer' 
    dataset = experiment.load_dataset(args)
    upfn = experiment.make_upfn(args, dataset, model, layername)
    sample_size = len(dataset)
    percent_level = 1.0 - args.quantile

    print('Inspecting layer %s of model %s on %s' % (layername, args.model, args.dataset))

    classlabels = dataset.classes

    ##### Broken library : need to rerun to load seglabels #######
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(25)

    try:
        segmodel, seglabels, segcatlabels = experiment.setting.load_segmenter(args.seg)
    except TimeOutException as ex:
        print(ex)
    signal.alarm(0)
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(25)

    try:
        segmodel, seglabels, segcatlabels = experiment.setting.load_segmenter(args.seg)
    except TimeOutException as ex:
        print(ex)
    signal.alarm(0)
    
    renorm = renormalize.renormalizer(dataset, target='zc')
    indices = [200, 755, 709, 423, 60, 100, 110, 120]
    batch = torch.cat([dataset[i][0][None,...] for i in indices])
    
    
    iv = imgviz.ImageVisualizer(120, source=dataset)
    seg = segmodel.segment_batch(renorm(batch).cuda(), downsample=4)
    
    pbar.descnext('rq')
    def compute_samples(batch, *args):
        image_batch = batch.cuda()
        _ = model(image_batch)
        acts = model.retained_layer(layername)
        hacts = upfn(acts)
        return hacts.permute(0, 2, 3, 1).contiguous().view(-1, acts.shape[1])
    rq = tally.tally_quantile(compute_samples, dataset,batch_size=4,
                              sample_size=sample_size,
                              r=8192,
                              num_workers=100,
                              pin_memory=True,
                              cachefile=resfile('rq.npz'))
    
    pbar.descnext('topk')
    def compute_image_max(batch, *args):
        image_batch = batch.cuda()
        _ = model(image_batch)
        acts = model.retained_layer(layername)
        acts = acts.view(acts.shape[0], acts.shape[1], -1)
        acts = acts.max(2)[0]
        return acts
    topk = tally.tally_topk(compute_image_max, dataset, sample_size=sample_size,
            batch_size=4, num_workers=30, pin_memory=True,
            cachefile=resfile('topk.npz'))
    
    level_at_99 = rq.quantiles(percent_level).cuda()[None,:,None,None]
    
    # Use the segmodel for segmentations.  With broden, we could use ground truth instead.
    def compute_conditional_indicator(batch, *args):
        image_batch = batch.cuda()
        seg = segmodel.segment_batch(renorm(image_batch), downsample=4)
        _ = model(image_batch)
        acts = model.retained_layer(layername)
        hacts = upfn(acts)
        iacts = (hacts > level_at_99).float() # indicator
        return tally.conditional_samples(iacts, seg)
    pbar.descnext('condi99')
    condi99 = tally.tally_conditional_mean(compute_conditional_indicator,
            dataset, sample_size=sample_size,batch_size=1,
            num_workers=3, pin_memory=True,
            cachefile=resfile('condi99.npz'))
    
    iou_99 = tally.iou_from_conditional_indicator_mean(condi99)
    unit_label_99 = [
            (concept.item(), seglabels[concept], segcatlabels[concept], bestiou.item())
            for (bestiou, concept) in zip(*iou_99.max(0))]
    label_list = [labelcat for concept, label, labelcat, iou in unit_label_99 if iou > 0.04]
    np.save(os.path.join(resdir, 'label_list.npy'), label_list)
    


    
    


