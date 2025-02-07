# instanseg-monkey-challenge.

# More details and trained models coming after the end of the competition!


Overall idea:
This solution first uses InstanSeg to detect all nuclei in the ROI. 

A pretrained InstanSeg model (brightfield_nuclei: Apache 2.0) was fine-tuned with the provided point annotations.

Then 128 by 128 patches at 0.5 microns/pixel are extracted to produce a 4-channel image (where the last channel is a binary mask of the nucleus of interest). 

We then give the patches to a efficientnet_l for classification into three classes (monocyte,lymphocyte or other).

For training we first train a small network (mobilenet4) to classify classes based on the IHC images (easy task, F1 score 0.86). We then generate a dataset of roughly 2 million annotations of PAS-CPG images based on the matched IHC image (we call this dataset "silver standard"). We train a large efficientnet_l on both the silver standard dataset and the smaller "gold standard" dataset provided. (F1 = 0.78 on silver and F1 = 0.76 on gold).

The final submission uses an ensemble of 4 models and 2 x 4 TTA augmentations. 


