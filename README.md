# Univariate-Time-Series-Prediction-using-Deep-Learning

### 0. Overview
This repository provides **Univariate Time Series Prediction**. It supports; 
- using various deep learning models including **DNN**, **CNN**, **RNN**, **LSTM**, **GRU**, and **Attention LSTM**.
- using **single-step** and **multi-step** prediction.

The dataset used is **Appliances Energy Prediction Data Set** and can be found [here](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction).

### 1. Quantitative Analysis

According to the table below, **CNN using 1D Convolutional layer** outperformed the other models. 
| Model | MAE↓ | MSE↓ | RMSE↓ | MPE↓ | MAPE↓ | R Squared↑ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DNN | 29.3038 | 3673.7921 | 57.0114 | -15.6800 | 26.4763 | 0.3820 | 
| **CNN** | **27.5182** | 3614.1634 | **56.1604** | **-11.2039** | **23.7301** | **0.4057** |
| RNN | 29.1327 | 3627.1491 | 56.7243 | -16.2193 | 26.9323 | 0.3809 |
| LSTM | 29.6157 | 3575.5541 | 56.4002 | -16.7178 | 27.9683 | 0.3771 | 
| GRU | 29.0402 | **3564.9701** | 56.2790 | -16.9984 | 26.9390 | 0.3872 |
| Attentional LSTM | 28.9658 | 3603.0751 | 56.3838 | -16.8199 | 26.3129 | 0.3898 |

| Model | MAE↓ | MSE↓ | RMSE↓ | MPE↓ | MAPE↓ | R Squared↑ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **DNN** | **31.3555** | 2913.6521 | **49.3946** | **-16.7329** | **29.1459** | **0.1775** |
| CNN | 32.9762 | **2893.2201** | 49.5900 | -21.7513 | 32.3016 | 0.1206 |
| RNN | 32.9153 | 2951.9055 | 50.0931 | -20.7460 | 32.2081 | 0.1223 |
| LSTM | 32.8141 | 2955.5278 | 50.1237 | -20.5471 | 32.0873 | 0.1191 |
| GRU | 33.0092 | 2927.5575 | 49.9503 | -21.2869 | 32.5345 | 0.1177 |
| Attentional LSTM | 32.2182 | 2920.8744 | 49.7972 | -19.1188 | 30.8223 | 0.1347 |

### 2. Qualitative Analysis
It definitely suffers from the typical lagging issue. Also, I averaged multi-step for plotting thus it looks to be smoothed.

<img src = './results/plots/Appliances Energy Prediction using AttentionalLSTM and SingleStep.png' width="500">
<img src = './results/plots/Appliances Energy Prediction using AttentionalLSTM and MultiStep.png' width="500">

### 3. Run the Codes

#### 1) Train 
If you want to train *Attention LSTM*, 

```
python main.py --model 'attention'
```

If you want to train with multi-step with time step of 5,

```
python main.py --model 'attention' --multi_step True --output_size 5
```

#### 2) Test
```
python main.py --model 'attention' --mode 'test'
```

To handle more arguments, you can refer to [here](https://github.com/hee9joon/Univariate-Time-Series-Prediction-using-Deep-Learning/blob/main/main.py#L255).


### Development Environment
```
- Windows 10 Home
- NVIDIA GFORCE RTX 2060
- CUDA 10.2
- torch 1.6.0
- torchvision 0.7.0
- etc
```
