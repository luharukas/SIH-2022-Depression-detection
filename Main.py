# importing all the libraries
import os
import sys
import nibabel as nib
import nipype as nip
import dipy as dip
import numpy as np
import pandas as pd
import nilearn as nil
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_dilation
from dipy.reconst import shm
from dipy.tracking import utils
from dipy.direction import peaks
#importing all the libraries to do preprocessing
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel,fractional_anisotropy,color_fa
from dipy.segment.mask import median_otsu
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response_ssst)
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.viz import window, actor, colormap, has_fury
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.segment.mask import median_otsu,applymask
from dipy.tracking.utils import density_map,target
from sklearn.cluster import KMeans


meta_data=pd.read_csv("C:/Users/luhar/RSCAMVITM/sihdata/Data3/meta_data.csv")

feature_data=[]

for i in range(len(meta_data)):
    dmri=meta_data['MRI_path'][i]
    fbvec=meta_data['bvec_path'][i]
    fbval=meta_data['bval_path'][i]
    label=meta_data['labels'][i]
    mask=meta_data['mask'][i]
    data,affine,img = load_nifti(dmri,return_img = True)
    data_label= load_nifti_data(label,)
    data_mask = load_nifti_data(mask,)
    stripped_data=applymask(data,data_mask)
    bvals, bvecs = read_bvals_bvecs(fbval,fbvec)
    gtab = gradient_table(bvals, bvecs)
    white_matter = binary_dilation((data_label == 2) | (data_label == 7 ) | (data_label==41) | (data_label==46))
    csamodel = shm.CsaOdfModel(gtab, 6)

    csapeaks = peaks.peaks_from_model(model=csamodel,data=data,sphere=peaks.default_sphere,relative_peak_threshold=.8,min_separation_angle=45,mask=white_matter)
    print(i)
    affine = np.eye(4)
    seeds = utils.seeds_from_mask(white_matter, affine, density=1)
    stopping_criterion = BinaryStoppingCriterion(white_matter)

    streamline_generator = LocalTracking(csapeaks, stopping_criterion, seeds,affine=affine, step_size=0.5)
    streamlines = Streamlines(streamline_generator)

    cc_slice = (data_label == 17) | (data_label==53)
    cc_streamlines = utils.target(streamlines, affine, cc_slice)
    cc_streamlines = Streamlines(cc_streamlines)

    other_streamlines = utils.target(streamlines, affine, cc_slice,
                                 include=False)
    other_streamlines = Streamlines(other_streamlines)
    assert len(other_streamlines) + len(cc_streamlines) == len(streamlines)

    M, grouping = utils.connectivity_matrix(cc_streamlines, affine,data_label.astype(np.uint8),return_mapping=True,mapping_as_streamlines=True)
    M[:3, :] = 0
    M[:, :3] = 0
    log_m=np.log1p(M)
    log_m = np.reshape(log_m,np.product(log_m.shape))
    feature_data.append(log_m)


    
kmeans = KMeans(n_clusters=2, random_state=0).fit(feature_data)




    