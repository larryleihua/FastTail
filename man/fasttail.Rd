\name{fasttail}
\alias{fasttail}
\title{Fast inference for bivariate tail dependence and tail asymmetry}
\usage{
fasttail(dat, model = c("GGEE", "PPPP"), random = c("quasi", "pseudo"),
  verbose = TRUE)
}
\arguments{
\item{dat}{n x 2 matrix or data frame of uniform scores; non-uniform input is
converted to uniform scores.}
\item{model}{GGEE or PPPP copula. Default GGEE.}
\item{random}{Sampling scheme used to train the NBE. Default quasi.}
\item{verbose}{Logical; report which NBE was selected.}
}
\value{
A list with \code{estimates} (alpha, beta) and \code{utd} (unified tail
dependence parameters, lower and upper).
}
\description{
Fast inference using bundled pre-trained neural Bayes estimators. Requires
only base R.
}
\examples{
dat <- subset(cobemo, copula == "bb7")[, c("u", "v")]
fasttail(dat)
}
\source{
Hua, L. (2026), Amortized Neural Inference on Bivariate Tail Dependence and
Tail Asymmetry.
}
