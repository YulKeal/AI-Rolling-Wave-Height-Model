## AI Rolling Wave Height Model
[![License](https://img.shields.io/static/v1?label=License&message=Apache&color=<Yellow>)](https://github.com/huggingface/diffusion-models-class/blob/main/LICENSE) &nbsp;
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)
[![Google Drive](https://img.shields.io/badge/GoogleDrive-WeightsFile-blue?logo=GoogleDrive)](https://drive.google.com/drive/folders/1BsIlwOFY8mV6CDcUPLvZXFXRDJ0c-TYk?usp=sharing)

A global-scale SWH roll forecasting model based on data assimilation techniques can be used

## Introduction

The global-scale SWH roll model is a simple but efficient predictive modeling framework that can accurately predict the SWH of the hour after the hour by inputting the SWH 
of the previous hour and the wind of the hour after the hour after the hour after the hour. as long as the wind field is required for the whole year, the model is able to quickly 
complete the modeling of SWH for the whole year, and the forecast error is not significantly larger than state-of-the-art numerical wave models, which can be used as a time- or 
computationally-limited modeling tool. case, a surrogate for a numerical model.
<p align="left">
  <img src="https://github.com/YulKeal/AI-Rolling-Wave-Height-Model/blob/main/figure/figure2.jpg" alt="" width="600"/>
</p>

Additionally in the rolling forecasting process, we integrate a data assimilation approach that aims to improve the forecast accuracy by integrating the observed data into the forecast process.
Conventional models usually rely on static initial conditions, which leads to an increase in error over time. This rolling model addresses this limitation by continuously correcting the model 
predictions based on real-time or periodic assimilation data. The implementation shows that the introduction of the assimilated rolling model not only improves the results dramatically, 
but also converges faster and reaches the steady state quickly.
<p align="left">
  <img src="https://github.com/YulKeal/AI-Rolling-Wave-Height-Model/blob/main/figure/figure3.jpg" alt="" width="600"/>
</p>

## DataSets

| DataSet Name                                                                      | Parameters Used|
|:------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
|[ERA5 hourly data on single levels from 1940 to present](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview)   | U10、V10、SWH|
| [CCI](https://archive.ceda.ac.uk/) |SWH Denoised|


## Results
From 2020-01-01 00:00 as the initial field, even without using data assimilation, rolling predictions for a full year of the 2020 test set works very well, and it is hard to tell the difference with the naked eye compared to ERA5
<p align="left">
  <img src="https://github.com/YulKeal/AI-Wave-Height-Model/blob/main/figure/Movies-S1.gif" alt="Global" width="500"/>
</p>


## How to Use
You can directly use the rolling_model.py file and call either NetTrain() for network training or NetInference() for inference. In the NetInference() function, you can modify the forecast_steps and Enable_Assi parameters in the continuous_inference function to choose the number of rolling forecast steps or whether to enable assimilation. The model weight files and 2020 CCI altimeter data (used for assimilation) are provided in Google Drive. The ERA5 2020 test set data we use is quite large, and you can download it yourself via API using the download_era5.py script (please refer to the official website for API key configuration). The assimilation.py file defines the implementation of the assimilation functionality.Additionally, mask_land.npz is a mask used to exclude land and ice regions.
