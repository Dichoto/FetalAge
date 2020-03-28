#%%
import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import torch
import nibabel as nib
#%%
def sensitivity_analysis(model, image_tensor, device, postprocess='abs'):
    """
    Perform sensitivity analysis (via backpropagation; Simonyan et al. 2014) to determine the relevance of each image pixel 
    for the classification decision. Return a relevance heatmap over the input image.
    
    Args:
        model (torch.nn.Module): The pytorch model. Should be set to eval mode. 
        image_tensor (torch.Tensor or numpy.ndarray): The image to run through the `model` (channels first!).
        target_class (int): The target output class for which to produce the heatmap. 
                      If `None` (default), use the most likely class from the `model`s output.
        postprocess (None or 'abs' or 'square'): The method to postprocess the heatmap with. `'abs'` is used 
                                                 in Simonyan et al. 2014, `'square'` is used in Montavon et al. 2018.
        
    Returns:
        A numpy array of the same shape as image_tensor, indicating the relevance of each image pixel. 
    """
    if postprocess not in [None, 'abs', 'square']:
        raise ValueError("postprocess must be None, 'abs' or 'square'")
    
    # Forward pass.
    X = torch.from_numpy(image_tensor)  # convert numpy or list to tensor
    X.unsqueeze_(0) # add channel of 1
    X.unsqueeze_(0) # mimic batch of 1
    X = X.to(device)
    X.requires_grad_()
    output = model(X)
    
    # Backward pass.
    model.zero_grad()
    output.backward()
        
    relevance_map = X.grad.data[0,0].cpu().numpy()
    
    # Postprocess the relevance map.
    if postprocess == 'abs':  # as in Simonyan et al. (2014)
        return np.abs(relevance_map)
    elif postprocess == 'square':  # as in Montavon et al. (2018)
        return relevance_map**2
    elif postprocess is None:
        return relevance_map

def SenAna_sub(model, img_sub, device, postprocess='abs'):
    """
    Perform sensitivity analysis at subject level, i.e. get average relevance map across all image frames.
    
    Args:
        model (torch.nn.Module): the pytorch CNN model. Must set to eval mode.
        igm_sub (4D numpy array of shape (n_frames, x, y, z)): the array of multiple images for one subject
        device: cuda or cpu
        postprocess: method for postprocessing heatmap.
        
    Return:
        a numpy array of the same shape with img_sub, which is the relevance map of each image frame
    """
    if postprocess not in [None, 'abs', 'square']:
        raise ValueError("postprocess must be None, 'abs' or 'square'")
    
    # Forward pass.
    img_shape = img_sub.shape
    relevance_map = np.zeros(img_shape)
    for i in range(img_shape[0]):
        X = torch.from_numpy(img_sub[i])  # convert numpy or list to tensor
        X.unsqueeze_(0) # add channel of 1
        X.unsqueeze_(0) # mimic batch of 1
        X = X.to(device)
        X.requires_grad_()
        output = model(X)
    
        # Backward pass.
        model.zero_grad()
        output.backward()
            
        relevance_map[i] = X.grad.data[:,0,:,:,:].cpu().numpy()
    
    # Postprocess the relevance map.
    if postprocess == 'abs':  # as in Simonyan et al. (2014)
        return np.abs(relevance_map)
    elif postprocess == 'square':  # as in Montavon et al. (2018)
        return relevance_map**2
    elif postprocess is None:
        return relevance_map

def areaSC(mask, relevance):
    n_mask = mask.shape[3]
    area_sc = []
    for i in range(n_mask):
        area_sc.append(np.sum(mask[:,:,:,i]*relevance))
    return np.array(area_sc)
    

