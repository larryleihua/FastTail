# FastTail

Fast inference on bivariate tail dependence and tail asymmetry using amortized
neural Bayes estimators (NBEs).

The pre-trained NBEs are bundled with the package, so inference needs **only
base R** — no Julia, no Flux, no GPU, no compilation.

## Installation

```r
# install.packages("remotes")
remotes::install_github("larryleihua/FastTail")
```

Requires R >= 4.0. There are no package dependencies.

## Quick examples

```r
library(FastTail)

dat1 <- subset(cobemo, copula == "bb7")[, c("u", "v")]
fasttail(dat1)

dat2 <- CopulaOne::rGGEE_COP(700, al = 1.4, be = 0.8)
fasttail(dat2)
```

`fasttail()` returns the estimated copula parameters (`alpha`, `beta` > 0) and
the unified tail dependence parameters (`lower`, `upper`, both in (0, 1); larger
means stronger dependence in that tail).

Arguments: `model` is `"GGEE"` (default) or `"PPPP"`; `random` is `"quasi"`
(default) or `"pseudo"`. The NBE trained at the nearest replicate count
m in {78, 195, 390, 780} is selected automatically from `nrow(dat)`.

## Notes

Data outside [0, 1] is converted to uniform scores automatically. Data already
in [0, 1] is used as given, so transform it yourself if it is not already on the
copula scale.

Sample sizes above 780 fall back to the m = 780 estimator, which is not ideal.

## References

- Hua, L. (2026), *Amortized neural inference on bivariate tail dependence and
  tail asymmetry*. [Full Text](https://www.degruyterbrill.com/document/doi/10.1515/demo-2025-0021/html)

## License

GPL-3
