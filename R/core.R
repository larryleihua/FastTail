#' Fast inference for bivariate tail dependence and tail asymmetry
#'
#' Fast inference for bivariate tail dependence and tail asymmetry. The first run is a bit slower as it needs to load the model, and after the first run the speed will be ultra fast.
#' @importFrom stats ks.test
#' @param dat input of uniform scores, should be n rows and 2 columns; if not uniform scores the data will be converted to uniform scores for further calculation.
#' @param model GGEE or PPPP copulas, default is GGEE copula
#' @param random Methods for generating random samples used for training NBE, default is quasi
#' @returns estimated parameters (alpha, beta > 0) of the copula, and the unified tail dependence parameters (0 < utd_lower, utd_upper < 1, the larger the stronger degree of dependence in the tails)
#' @source Based on Hua, L. (2025), Amortized Neural Inference on Bivariate Tail Dependence and Tail Asymmetry
#' @keywords tail dependence, intermediate tail dependence, copula
#' @examples
#' \dontrun{
#' dat <- subset(cobemo, copula=="bb7")[, c("u","v")]
#' fasttail(dat)
#' }
#' @export
fasttail <- function(dat, model="GGEE", random="quasi")
{
  if (ncol(dat) != 2) stop("The data should be n rows and 2 columns!")
  n <- nrow(dat)

  if(ks.test(dat[,1], "punif", min=0, max=1)$p.value < 0.05 |
     max(dat[,1]) > 1 | min(dat[,1]) < 0)
  {
    uscore <- (1:n - 0.5)/n
    cat("The 1st column is not uniformly distributed, and is now
        converted to uniform scores.", "\n")
    dat[,1] <- uscore[rank(dat[,1])]
  }

  if(ks.test(dat[,2], "punif", min=0, max=1)$p.value < 0.05 |
     max(dat[,2]) > 1 | min(dat[,2]) < 0)
  {
    uscore <- (1:n - 0.5)/n
    cat("The 2nd column is not uniformly distributed, and is now
        converted to uniform scores.", "\n")
    dat[,2] <- uscore[rank(dat[,2])]
  }

  JuliaConnectoR::juliaEval('using NeuralEstimators, Flux, CUDA, cuDNN')

  if(n <= 78)
  {
    m <- 78
    cat("The sample size is", n, "and NBE (m=78) is used", "\n")
  }else if(n <= 195)
  {
    m <- 195
    cat("The sample size is", n, "and NBE (m=195) is used", "\n")
  }else if(n <= 390)
  {
    m <- 390
    cat("The sample size is", n, "and NBE (m=390) is used", "\n")
  }else if(n <= 780)
  {
    m <- 780
    cat("The sample size is", n, "and NBE (m=780) is used", "\n")
  }else
  {
    m <- 780
    cat("The sample size is larger than 780, and m=780 will be used which is not ideal!", "\n")
  }

  nbe <- JuliaConnectoR::juliaEval('
  d = 2    # dimension of each replicate
  w = 32   # number of neurons in each hidden layer
  # Layer to ensure valid estimates
  final_layer = Parallel(
      vcat,
      Dense(w, 1, softplus), # alpha > 0
      Dense(w, 1, softplus)  # beta > 0
    )
  psi = Chain(Dense(d, w, relu), Dense(w, w, relu))
  phi = Chain(Dense(w, w, relu), final_layer)
  deepset = DeepSet(psi, phi)
  estimator = PointEstimator(deepset)
  ')

  nbe_weights <- system.file("nbe", paste(model, random, paste0("m", format(m), ".bson"), sep="_"), package = "FastTail")

  tryCatch({
    nbe <- NeuralEstimators::loadstate(nbe, nbe_weights)
  }, error = function(e) {
    message(sprintf("The NBE cannot be loaded!"))
  })

  estimated_parameters <- t(NeuralEstimators::estimate(nbe, t(dat)))
  al <- estimated_parameters[1]
  be <- estimated_parameters[2]
  utd_lower <- atan(1/al)*2/pi
  utd_upper <- atan(1/be)*2/pi
  estimates <- data.frame(alpha=al,beta=be)
  rownames(estimates) <- paste0("Estimated parameters (", model, "):")
  unified_tail_dependence <- data.frame(lower=utd_lower,upper=utd_upper)
  rownames(unified_tail_dependence) <- paste0("Unified Tail Dependence Parameters (", model, ", between 0 and 1):")
  return(list(estimates=estimates, utd=unified_tail_dependence))
}

