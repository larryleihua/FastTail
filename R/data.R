#' The Copula Benchmark Data (Moderate dependence)
#'
#' This benchmark data can be used to compare modeling capacity of different copula models when the model is mis-specified.
#'
#' @format A data frame with 9360 rows and 3 variables:
#' \describe{
#'   \item{u}{the first uniform scores}
#'   \item{v}{the second uniform scores}
#'   \item{copula}{copula families, including "gaussian", "t", "clayton", "gumbel", "frank", "joe", "bb1", "bb6", "bb7", "bb8", "GGEE", "PPPP"}
#' }
#' @source Simulated and used in the paper: Hua, L. (2025), Amortized Neural Inference on Bivariate Tail Dependence and Tail Asymmetry
#' @examples
#' data(cobemo)
#' head(cobemo)
"cobemo"
