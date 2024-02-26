# ClimODE: Climate and Weather Forecasting With Physics-informed Neural ODEs
The code repository for the paper ClimODE: Climate and Weather Forecasting With Physics-informed Neural ODEs. More information can be found on the project [website](https://yogeshverma1998.github.io/ClimODE/). 
<p align="center">
  <img src="https://github.com/Aalto-QuML/ClimODE/blob/main/workflow_final_climate_v6.png" />
</p>

## Citation
If you find this repository useful in your research, please consider citing the following paper:
 ```
@inproceedings{
anonymous2024climode,
title={Clim{ODE}: Climate Forecasting With Physics-informed Neural {ODE}s},
author={Yogesh Verma, Markus Heinonen, Vikas Garg},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=xuY33XhEGR}
}

```

## Prerequisites

- torchdiffeq : https://github.com/rtqichen/torchdiffeq.
- pytorch >= 1.12.0
- torch-scatter 
- torch-sparse 
- torch-cluster 
- torch-spline-conv 
- torchcubicspline: https://github.com/patrick-kidger/torchcubicspline

## Data Preparation

First, download ERA5 data with 5.625deg from [WeatherBench](https://dataserv.ub.tum.de/index.php/s/m1524895). The data directory should look like the following
```
era5_data
   |-- 10m_u_component_of_wind
   |-- 10m_v_component_of_wind
   |-- 2m_temperature
   |-- constants
   |-- geopotential_500
   |-- temperature_850
```

## Training

### Global Forecast

To train ClimODE for global forecast use,

```
python train_global.py --scale 0 --batch_size 8 --spectral 0 --solver "euler" 
```

### Regional Forecast

To train ClimODE for regional forecast among various regions of earth use,
```
python train_region.py --scale 0 --batch_size 8 --spectral 0 --solver "euler" --region 'NorthAmerica/SouthAmerica/Australia'
```

## Evaluation

### Global Forecast

To evaluate ClimODE for global forecast on Lat. weighted RMSE and ACC use, (Make sure to change the model path in the file)

```
python evaluation_global.py --spectral 0 --scale 0 
```

To evaluate ClimODE for regional forecast on Lat. weighted RMSE and ACC use, (Make sure to change the model path in the file)

```
python evaluation_region.py --spectral 0 --scale 0 --region 'NorthAmerica/SouthAmerica/Australia'
```



