## One-time: flat binary -> R/sysdata.rda (internal, lazy-loaded).
## Run from the repo root, after data-raw/export_weights.py:
##     Rscript data-raw/build_sysdata.R
## Only needed when the NBEs are retrained. Users never run this.

idx <- read.csv("data-raw/weights_index.csv", stringsAsFactors = FALSE)
v   <- readBin("data-raw/weights.bin", "double",
               n = sum(idx$n), size = 8)

## Julia and R are both column-major, so no transposition is needed.
.nbe <- lapply(split(idx, idx$model), function(m) {
  out <- lapply(seq_len(nrow(m)), function(i) {
    x <- v[(m$off[i] + 1):(m$off[i] + m$n[i])]
    if (m$d2[i] > 0) matrix(x, nrow = m$d1[i], ncol = m$d2[i]) else x
  })
  names(out) <- m$param
  out
})

save(.nbe, file = "R/sysdata.rda", compress = "xz")
cat("wrote R/sysdata.rda:", length(.nbe), "models,",
    file.size("R/sysdata.rda"), "bytes\n")
