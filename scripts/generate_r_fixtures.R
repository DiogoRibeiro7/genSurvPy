# Frozen reference outputs from the R `genSurv` package.
#
# Four generators were ported from R -- genCPHM, genCMM, genTHMM and genTDCM --
# and `tests/test_r_parity.py` checks that ours still agree with them in
# distribution. R and NumPy use different generators (Mersenne Twister against
# PCG64), so identical draws are impossible and parity is on distributions, not
# values.
#
# Regenerate with:
#
#     Rscript scripts/generate_r_fixtures.R
#
# It writes gzipped CSVs into tests/fixtures/r_parity/. Commit them: CI has no
# R, and the point of freezing is that the reference does not move underneath
# the comparison. Record the genSurv version when you do -- it is written to
# VERSION.txt beside the fixtures.
#
# Sizes are chosen so the Monte Carlo error on the compared statistics is small
# next to a divergence worth catching, while keeping the committed files small.

suppressWarnings(suppressMessages(library(genSurv)))

out_dir <- file.path("tests", "fixtures", "r_parity")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_fixture <- function(data, name) {
  # Eight significant digits. The comparisons are distributional -- a
  # Kolmogorov-Smirnov statistic and a handful of rates -- so full double
  # precision would only make the committed files larger.
  numeric_columns <- vapply(data, is.numeric, logical(1))
  data[numeric_columns] <- lapply(data[numeric_columns], signif, digits = 8)

  path <- file.path(out_dir, paste0(name, ".csv.gz"))
  connection <- gzfile(path, "w")
  write.csv(data, connection, row.names = FALSE)
  close(connection)
  cat(sprintf("  %-8s %6d rows -> %s\n", name, nrow(data), path))
}

# Every call fixes its own seed, so one fixture can be regenerated without
# moving the others.

set.seed(20260828)
write_fixture(
  genCPHM(n = 20000, model.cens = "uniform", cens.par = 1, beta = 0.5, covar = 2),
  "cphm"
)

set.seed(20260829)
write_fixture(
  genCMM(
    n = 25000, model.cens = "uniform", cens.par = 1,
    beta = c(0.1, 0.2, 0.3), covar = 1,
    rate = c(0.1, 1, 0.2, 1, 0.1, 1)
  ),
  "cmm"
)

set.seed(20260830)
write_fixture(
  genTHMM(
    n = 25000, model.cens = "uniform", cens.par = 1,
    beta = c(0.1, 0.2, 0.3), covar = 1,
    rate = c(0.2, 0.3, 0.4)
  ),
  "thmm"
)

set.seed(20260831)
write_fixture(
  genTDCM(
    n = 20000, dist = "weibull", corr = 0.5, dist.par = c(1, 2, 1, 2),
    model.cens = "uniform", cens.par = 1, beta = c(0.5, 0.3), lambda = 1
  ),
  "tdcm"
)

writeLines(
  c(
    paste0("genSurv ", as.character(packageVersion("genSurv"))),
    paste0("R ", paste0(R.version$major, ".", R.version$minor)),
    paste0("generated ", format(Sys.Date()))
  ),
  file.path(out_dir, "VERSION.txt")
)

cat("\nDone. Parameters live in this script; the Python side reads them from\n")
cat("tests/test_r_parity.py, which must be kept in step with it.\n")
