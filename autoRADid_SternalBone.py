# -*- coding: utf-8 -*-
"""

autoRADid_SternalBone
Automatic radiologic identification pipeline based on sternal bone
Dominique Neuhaus
November 2022

"""


# %% Pkgs/Libs
import numpy as np
import pandas as pd
import os
import cv2 as cv
from medpy.metric.binary import jc
from medpy.metric.binary import dc
from medpy.metric.image import mutual_information as mi
import SimpleITK as sitk



# %% Pre-Processing PM image (of unknown deceased)
# Import PM Image (MIP Screenshot)--------------------------------------------------------------------------
ID =        # Define name of your file
Unknown =   # Define name of base image
PM_dir =    # Define path with PM data
Path = PM_dir + Unknown
PM_img = cv.imread(Path, 0)


# Apply Otsu on PM Image (-> OpenCV) ------------------------------------------------------------------------------------
ret,PM_img_Otsu = cv.threshold(PM_img,120,255,cv.THRESH_BINARY+cv.THRESH_OTSU)   #120, 255


# Removing smaller Blobs: cv.connectedComponentsWithStats (-> OpenCV)-------------------------------------------------------
nb_blobs, im_with_separated_blobs, stats, _ = cv.connectedComponentsWithStats(PM_img_Otsu)

sizes = stats[:, -1]

sizes = sizes[1:]
nb_blobs -= 1

min_size = 800  # vary to achieve satisfying results. Removing small parts while keeping sternal bone

# output image with only the kept components
PM_img_Blotsu = np.zeros((PM_img_Otsu.shape))
# for every component in the image, keep it only if it's above min_size
for blob in range(nb_blobs):
	if sizes[blob] >= min_size:
		PM_img_Blotsu[im_with_separated_blobs == blob + 1] = 255
                
        
# Convert array to SimpleITK.Image-------------------------------------------------------
Conv_Itk_PM = sitk.GetImageFromArray(PM_img_Blotsu)
 


# %% Looping through AM Pool - Preprocessing, Registration, and Similarity

# Directory with AM images (MIP Screenshots) and Preparing Results Array
AM_dir =    # Define path with AM data

PathSaveReg =   # Define path to save registered AM files to

No_of_files = len(os.listdir(AM_dir))+1     #+1 to make up for header line 
Results = np.empty(shape=(No_of_files,3),dtype='object')
Results[0,1] = 'Jaccard'
Results[0,2] = 'Dice'

i = 1
j = 0


for filename in os.listdir(AM_dir):
	img_path = os.path.join(AM_dir, filename)
	data_img = cv.imread(img_path,0)
    
    #Threshold Otsu (-> OpenCV)
	ret2,AM_img_Otsu = cv.threshold(data_img,120,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
	
    #Removing Blobs (-> OpenCV)
	nb_blobs, im_with_separated_blobs, stats, _ = cv.connectedComponentsWithStats(AM_img_Otsu)

	sizes = stats[:, -1]

	sizes = sizes[1:]
	nb_blobs -= 1

	min_size = 800  # vary to achieve satisfying results. Removing small parts while keeping sternal bone


	AM_img_Blotsu = np.zeros((AM_img_Otsu.shape))

	for blob in range(nb_blobs):
		if sizes[blob] >= min_size:

			AM_img_Blotsu[im_with_separated_blobs == blob + 1] = 255
           
            
    #Converting array to SimpleITK.Image
	Conv_Itk_AM = sitk.GetImageFromArray(AM_img_Blotsu)
    

    #Registration (-> SimpleElastix)
	resultImage1 = sitk.Elastix(Conv_Itk_PM, Conv_Itk_AM, "translation")
	resultImage2 = sitk.Elastix(Conv_Itk_PM, resultImage1, "affine")
	Conv_arr_AM = sitk.GetArrayFromImage(resultImage2)
    
	cv.imwrite(PathSaveReg + filename, Conv_arr_AM)
	
    #Measure Similarity (-> medpy)
	RegAM_Img = cv.imread(PathSaveReg + filename,0)
	SimiJack = jc(PM_img_Blotsu, RegAM_Img)
	SimiDice = dc(PM_img_Blotsu, RegAM_Img)
	SimiMutI = mi(PM_img_Blotsu, RegAM_Img)
	Results[i,j] = filename
	Results[i,j+1] = SimiJack
	Results[i,j+2] = SimiDice

	
	i = i+1


# %% Save results in .csv file
df = pd.DataFrame(Results)
ResultsName = 'Results_' + ID + '.csv'
df.to_csv(ResultsName)


# %% Get Jaccard Values
# Make vector with all file names
AM_Names = Results[1:,0]

# Find highest Jaccard Coefficient and its location
JC_Results = Results[1:, 1] #First line is header
maxJC = np.amax(JC_Results)
PosMaxJC = np.where(JC_Results == maxJC)
PosMaxJC_df = pd.DataFrame(PosMaxJC).values       #get actual value to pass as index for Names

# Find matching file according to Jaccard
JCMatchFile = AM_Names[PosMaxJC_df[0,0]]

# Find position of values above 97% of matching value
JC_Res_ProC = 100/maxJC*JC_Results
Pos97JC = np.where(JC_Res_ProC >= 97)


# Get names of >= 97% AM files (according to Jaccard)
Candidates97JC = AM_Names[Pos97JC]


# %% Get Dice Values
# Find highest Dice Coefficient and its location
DC_Results = Results[1:, 2] #First line is header
maxDC = np.amax(DC_Results)
PosMaxDC = np.where(DC_Results == maxDC)
PosMaxDC_df = pd.DataFrame(PosMaxDC).values

DCMatchFile = AM_Names[PosMaxDC_df[0,0]]

DC_Res_ProC = 100/maxDC*DC_Results
Pos97DC = np.where(DC_Res_ProC >= 97)

Candidates97DC = AM_Names[Pos97DC]


# %% Prepare candidates for MI evaluation
# Summarising candidates97DC and -JC without doubles
Candidates97JC_ls = list(Candidates97JC)
Candidates97DC_ls = list(Candidates97DC)

DiffCandids = set(Candidates97DC_ls) - set(Candidates97JC_ls)

Candidates97ALL = Candidates97JC_ls + list(DiffCandids)


# %% Loop through all candidates and do MI
v = 0
MI_Results = np.zeros(shape=(1,1)) # dtype default is float64

for a in Candidates97ALL:
	PathRegAM = PathSaveReg + Candidates97ALL[v]
	RegAM_Img = cv.imread(PathRegAM,0)
	SimiMutI = mi(PM_img_Blotsu, RegAM_Img)
	MI_Results = np.append(MI_Results, SimiMutI)
    
	v = v+1
	
MI_Results = MI_Results[1:]   # First entry is zero according to allocation; must be skipped
maxMI = np.amax(MI_Results)
PosMaxMI = np.where(MI_Results == maxMI)

PosMaxMI_df = pd.DataFrame(PosMaxMI).values

MIMatchFile = Candidates97ALL[PosMaxMI_df[0,0]]

MI_Res_ProC = 100/maxMI*MI_Results
MI_Res_ProC = MI_Res_ProC[MI_Res_ProC != 100]   # 100% is the "real" match, shall not be considered when looking for close ones
Pos97MI = np.where(MI_Res_ProC >= 97)

Candidates97MI = AM_Names[Pos97MI]
Candidates97MI_ls = list(Candidates97MI)
emptyOrNot = len(Candidates97MI)


# Result Messages
if DCMatchFile == JCMatchFile:
	if MIMatchFile == DCMatchFile and emptyOrNot == 0:   #DCMatchFile == JCMatchFile, thus any can be taken
		print("The most probable match is" + " " + MIMatchFile)
	elif MIMatchFile == DCMatchFile and emptyOrNot != 0:   #DCMatchFile == JCMatchFile, thus any can be taken
		print("The most probable match is" + " " + MIMatchFile)
		print("Potentially other matching files are:")
		print(Candidates97MI_ls)
	elif MIMatchFile != DCMatchFile and emptyOrNot == 0:
		print("The match could not be confirmed. The two most probable ones are" + " " + DCMatchFile + " and " + MIMatchFile)
		print("Reasons for uncertainty may be surgical interventions on sternum and/or suboptimal quality of CT scan.")    
	elif MIMatchFile != DCMatchFile and emptyOrNot != 0:
		print("The match could not be confirmed. The two most probable ones are" + " " + DCMatchFile + " and " + MIMatchFile)
		print("Potentially other matching files are:")
		print(Candidates97MI_ls)
		print("Reasons for uncertainty may be surgical interventions on sternum and/or suboptimal quality of CT scan.")    
else:
	print("No match could be found.")




