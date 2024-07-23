# Basternid

![BasternidLogo](https://user-images.githubusercontent.com/114567200/204021043-cf59ebdd-eda6-4304-98e7-422f575fe94a.svg)

### The Basel Sternal Bone Identification Tool

Basternid is an automated radiologic identification tool to identify unknown deceased via comparison of post mortem with ante mortem computed tomography data. 
The anatomical structure used for identification is the sternal bone.

Created by the [Forensic Medicine and Imaging Research Group](https://dbe.unibas.ch/en/research/imaging-modelling-diagnosis/forensic-medicine-imaging-research-group/).
If you use it, please cite our publication: 
Neuhaus, D., Wittig, H., Scheurer, E. & Lenz, C. Fully automated radiologic identification focusing on the sternal bone. Forensic Sci. Int. 346, 111648 (2023).

# Requirements
+ Numpy
+ Pandas
+ os
+ OpenCV
+ MedPy
+ SimpleITK

# Pipeline
+ In a DICOM viewer, import the CT files and create the "base images": orthogonal view on the sternal bone.
+ Locate the ante mortem base images and the post mortem base images in two seperate folders.
+ The base images will be converted to binary images and small areas not belonging to the sternal bone will be removed.
+ The ante mortem data will be registered to the post mortem data to make up for different scaling and/or perspectives.
+ Eventually, ante mortem data will be compared with post mortem data using three different similarity measrues (Jaccard coefficient, Dice coefficient, and Mutual Information).
+ Returned will be the files with the highest similarity values, indicating the most probable match.

+ The pipeline performs a comparison of a large ante mortem data set with one post mortem image. However, by simply changing the location of the files, a large post mortem 
data set can be compared with one ante mortem file.


# Usage
+ Download the python script.
+ Define path where you have located your ante mortem and post mortem data, respectively. 
+ Define path where you want to save the registered ante mortem data.

# MIT License
