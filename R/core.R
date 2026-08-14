## Internal helpers ---------------------------------------------------------

relu <- function(x) pmax(x, 0)

## numerically stable log(1 + exp(x))
softplus <- function(x) pmax(x, 0) + log1p(exp(-abs(x)))

## DeepSet forward pass.
##   Z : 2 x m matrix of uniform scores (columns are replicates)
## Architecture (see Hua, 2025):
##   psi = Chain(Dense(2, 32, relu), Dense(32, 32, relu))
##   aggregation = mean over replicates
##   phi = Chain(Dense(32, 32, relu),
##               Parallel(vcat, Dense(32, 1, softplus), Dense(32, 1, softplus)))
nbe_forward <- function(w, Z) {
  h <- relu(w$psi1_W %*% Z + w$psi1_b)
  h <- relu(w$psi2_W %*% h + w$psi2_b)
  T <- rowMeans(h)
  g <- relu(drop(w$phi1_W %*% T) + w$phi1_b)
  c(alpha = softplus(drop(w$outA_W %*% g) + w$outA_b),
    beta  = softplus(drop(w$outB_W %*% g) + w$outB_b))
}

## Exported ------------------------------------------------------------------

#' Fast inference for bivariate tail dependence and tail asymmetry
#'
#' Fast inference for bivariate tail dependence and tail asymmetry using
#' pre-trained neural Bayes estimators (NBEs). The NBE weights are bundled with
#' the package, so no training, no GPU and no external runtime are required.
#'
#' @param dat input of uniform scores, should be n rows and 2 columns; if not
#'   uniform scores the data will be converted to uniform scores for further
#'   calculation.
#' @param model GGEE or PPPP copulas, default is GGEE copula
#' @param random Methods for generating random samples used for training NBE,
#'   default is quasi
#' @param verbose logical; print which NBE is selected. Default TRUE.
#' @returns estimated parameters (alpha, beta > 0) of the copula, and the
#'   unified tail dependence parameters (0 < utd_lower, utd_upper < 1, the
#'   larger the stronger degree of dependence in the tails)
#' @source Based on Hua, L. (2025), Amortized Neural Inference on Bivariate
#'   Tail Dependence and Tail Asymmetry
#' @keywords tail dependence, intermediate tail dependence, copula
#' @examples
#' dat <- subset(cobemo, copula == "bb7")[, c("u", "v")]
#' fasttail(dat)
#' @export
fasttail <- function(dat, model = c("GGEE", "PPPP"),
                     random = c("quasi", "pseudo"), verbose = TRUE) {
  model  <- match.arg(model)
  random <- match.arg(random)

  dat <- as.matrix(dat)
  if (ncol(dat) != 2) stop("The data should be n rows and 2 columns!")
  if (anyNA(dat))     stop("The data contain missing values.")
  n <- nrow(dat)
  if (n < 2) stop("At least 2 observations are required.")

  for (j in 1:2) {
    if (max(dat[, j]) > 1 || min(dat[, j]) < 0) {
      if (verbose)
        message("Column ", j, " is not uniformly distributed, and is now ",
                "converted to uniform scores.")
      dat[, j] <- (rank(dat[, j], ties.method = "average") - 0.5) / n
    }
  }

  m <- if (n <= 78) 78L else if (n <= 195) 195L else if (n <= 390) 390L else 780L
  if (verbose) {
    if (n > 780)
      message("The sample size is larger than 780, and m=780 will be used ",
              "which is not ideal!")
    else
      message("The sample size is ", n, " and NBE (m=", m, ") is used")
  }

  key <- paste(model, random, paste0("m", m), sep = "_")
  w   <- .nbe[[key]]
  if (is.null(w)) stop("No bundled NBE named '", key, "'.")

  est <- nbe_forward(w, t(dat))
  al  <- unname(est["alpha"])
  be  <- unname(est["beta"])

  estimates <- data.frame(alpha = al, beta = be)
  rownames(estimates) <- paste0("Estimated parameters (", model, "):")
  utd <- data.frame(lower = atan(1 / al) * 2 / pi,
                    upper = atan(1 / be) * 2 / pi)
  rownames(utd) <- paste0("Unified Tail Dependence Parameters (", model,
                          ", between 0 and 1):")
  list(estimates = estimates, utd = utd)
}
