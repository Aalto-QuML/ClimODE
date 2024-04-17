# ClimODE: Climate and Weather Forecasting With Physics-informed Neural ODEs

 [Yogesh verma](https://yoverma.github.io/yoerma.github.io/) | [Markus Heinonen](https://users.aalto.fi/~heinom10/) |  [Vikas Garg](https://www.mit.edu/~vgarg/)
 
The code repository for the paper ClimODE: Climate and Weather Forecasting With Physics-informed Neural ODEs. More information can be found on the project [website](https://yogeshverma1998.github.io/ClimODE/). 
<p align="center">
  <img src="https://github.com/Aalto-QuML/ClimODE/blob/main/workflow_final_climate_v6.png" />
</p>

## Citation
If you find this repository useful in your research, please consider citing the following paper:
 ```
@inproceedings{
verma2024climode,
title={Clim{ODE}: Climate Forecasting With Physics-informed Neural {ODE}s},
author={Yogesh Verma and Markus Heinonen and Vikas Garg},
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
- properscoring (for CRPS scores) : https://pypi.org/project/properscoring/

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

## Training ERA5

### Global Forecast

To train ClimODE for global forecast use,

```
python train_global.py --scale 0 --batch_size 8 --spectral 0 --solver "euler" 
```

### Global Monthly Forecast

To train ClimODE for global monthly forecast use,

```
python train_monthly.py --scale 0 --batch_size 4 --spectral 0 --solver "euler" 
```


### Regional Forecast

To train ClimODE for regional forecasts among various regions of earth use,
```
python train_region.py --scale 0 --batch_size 8 --spectral 0 --solver "euler" --region 'NorthAmerica/SouthAmerica/Australia'
```

## Evaluation ERA5

### Global Forecast

To evaluate ClimODE for global forecast on Lat. weighted RMSE and ACC use, (Make sure to change the model path in the file)

```
python evaluation_global.py --spectral 0 --scale 0 --batch_size 8 
```

### Global Monthly Forecast

To evaluate ClimODE for global monthly forecast on Lat. weighted RMSE and ACC use, (Make sure to change the model path in the file)

```
python evaluation_monthly.py --spectral 0 --scale 0 --batch_size 4 
```

### Regional Forecast

To evaluate ClimODE for regional forecast on Lat. weighted RMSE and ACC use, (Make sure to change the model path in the file)

```
python evaluation_region.py --spectral 0 --scale 0 --region 'NorthAmerica/SouthAmerica/Australia' --batch_size 8 
```

## Training on a different custom dataset

To train on a custom dataset, you need to follow the below guidelines

- **Data Loading**: You might want to change the data loading scheme depending on your data (e.g. seasonal, daily, etc., and with many different input channels), which can be found in ```utils.py``` in the data-loading function.
- **Fitting initial velocity**: Depending on the data, you need to estimate the initial velocity to train and test the model (For more details, see the manuscript) via the ```fit_velocity``` function. 
- **Model Function**: Depending on the input observable quantities, you might need to modify the number of input channels to model function in ```model_function.py```.
- **Training and evaluation**: Depending on your dataset, you might want to fine-tune and change the various hyper-parameters in training and evaluation files. Make sure to make them consistent in both of them. Also, we report CRPS scores for global hourly forecast only, if you want to compute them for every task please include the ```evaluation_crps_mm``` function.




