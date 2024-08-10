## AI Wave Height Model
[![License](https://img.shields.io/static/v1?label=License&message=Apache&color=<Yellow>)](https://github.com/huggingface/diffusion-models-class/blob/main/LICENSE) &nbsp;
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-WeightsFile-blue?logo=HuggingFace)](https://huggingface.co/YulKeal/AI-Wave-Height-Model/tree/main)

A global-scale SWH roll forecasting model based on data assimilation techniques can be used

## Introduction
The global-scale SWH roll model is a simple but efficient predictive modeling framework that can accurately predict the SWH of the hour after the hour by inputting the SWH 
of the previous hour and the wind of the hour after the hour after the hour after the hour. as long as the wind field is required for the whole year, the model is able to quickly 
complete the modeling of SWH for the whole year, and the prediction is basically comparable to the effect of the state-of-the-art numerical models, which can be used as a time- or 
computationally-limited modeling tool. case, a surrogate for a numerical model.

Additionally in the rolling forecasting process, we integrate a data assimilation approach that aims to improve the forecast accuracy by integrating the observed data into the simulation process.
Conventional models usually rely on static initial conditions, which leads to an increase in error over time. This rolling model addresses this limitation by continuously correcting the model 
predictions based on real-time or periodic assimilation data. The implementation shows that the introduction of the assimilated rolling model not only improves the results dramatically, 
but also converges faster and reaches the steady state quickly.

## DataSets

| DataSet Name                                                                      | Parameters Used|
|:------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
|[ERA5 hourly data on single levels from 1940 to present](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview)   | U10、V10、SWH|
| [CCI](https://data.marine.copernicus.eu/product/BLKSEA_ANALYSISFORECAST_WAV_007_003/description) |SWH Denoised|


## Results



