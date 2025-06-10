## What is FastTail?
FastTail is dedicated for fast inference on tail dependence and tail asymmetry using amortized neural inference methods.

## System preparation

### Install Julia
- Install Julia from [here](https://julialang.org/install/)
- Run julia in your terminal
- Install packages in Julia: press ']' to switch to the package manager (Pkg) mode, then 
```
add NeuralEstimators@0.1.2 Flux@0.16.3 CUDA@5.6.1, cuDNN@1.4.1, BSON@0.3.9
```
- Install FastTail
```r
devtools::install_github("larryleihua/FastTail", force=T)
```

### The following systems were tested
- Windows-11 / Ubuntu 22.04, Julia-1.11.2, R-4.3.3
- Julia packages: NeuralEstimators-0.1.2, Flux-0.16.3, CUDA-5.6.1, cuDNN-1.4.1, BSON-0.3.9
- R packages: NeuralEstimators-0.2.0; JuliaConnectoR-1.1.4 

## Quick examples
```r
library(FastTail)
dat1 <- subset(cobemo, copula=="bb7")[,c("u","v")]
fasttail(dat1)

dat2 <- CopulaOne::rGGEE_COP(700, al=1.4, be=0.8)
fasttail(dat2)
```