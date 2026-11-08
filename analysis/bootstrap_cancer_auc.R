setwd(commandArgs(trailingOnly = TRUE)[1])

library(Rcpp)

seeds <- c(101, 202, 303, 404, 505)
folds <- 0:4
n_bootstrap <- 1000L
bootstrap_seed <- 20260904L
horizons <- c(`1 year` = 365.25, `3 years` = 1095.75, `5 years` = 1826.25)

oof_root <- "Inputs/oof"
master_file <- "Inputs/master_folds.csv"
old_summary_file <- "Source_Data/ExtendedData1/EDFig2cd_cancer_specific_td_auc_summary.tsv"
output_directory <- "Source_Data/ExtendedData1"
dir.create(output_directory, recursive = TRUE, showWarnings = FALSE)

Rcpp::cppFunction('
double weighted_auc_cpp(
    NumericVector risk,
    NumericVector case_weight,
    IntegerVector control) {
  int n = risk.size();
  std::vector<double> control_risk;
  control_risk.reserve(n);
  double total_case_weight = 0.0;
  for (int i = 0; i < n; ++i) {
    if (control[i] == 1) control_risk.push_back(risk[i]);
    if (case_weight[i] > 0.0) total_case_weight += case_weight[i];
  }
  if (control_risk.empty() || total_case_weight <= 0.0) return NA_REAL;
  std::sort(control_risk.begin(), control_risk.end());

  double numerator = 0.0;
  for (int i = 0; i < n; ++i) {
    if (case_weight[i] <= 0.0) continue;
    std::vector<double>::iterator lower = std::lower_bound(
      control_risk.begin(), control_risk.end(), risk[i]);
    std::vector<double>::iterator upper = std::upper_bound(
      control_risk.begin(), control_risk.end(), risk[i]);
    numerator += case_weight[i] * (
      lower - control_risk.begin() + 0.5 * (upper - lower));
  }
  return numerator / (total_case_weight * control_risk.size());
}
')

as_flag <- function(x) tolower(as.character(x)) == "true"

estimate_censoring_distribution <- function(time, event) {
  unique_times <- sort(unique(time))
  n_at_risk <- vapply(unique_times, function(value) sum(time >= value), numeric(1))
  deaths <- vapply(unique_times, function(value) sum(time == value & event), numeric(1))
  censored <- vapply(unique_times, function(value) sum(time == value & !event), numeric(1))
  risk_set <- n_at_risk - deaths
  hazard <- ifelse(censored == 0, 0, censored / risk_set)
  list(time = unique_times, survival = cumprod(1 - hazard))
}

predict_censoring_survival <- function(fit, time) {
  index <- findInterval(time, fit$time)
  output <- rep(1, length(time))
  output[index > 0] <- fit$survival[index[index > 0]]
  output
}

read_seed_oof <- function(seed) {
  frames <- lapply(folds, function(fold) {
    frame <- read.delim(
      file.path(
        oof_root, paste0("seed_", seed), paste0("fold", fold),
        "survival_oof_predictions.tsv.gz"
      ),
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
    frame$event <- as_flag(frame$event)
    frame$complete_case <- as_flag(frame$complete_case)
    frame
  })
  do.call(rbind, frames)
}

seed_oof <- setNames(lapply(seeds, read_seed_oof), as.character(seeds))
reference_patients <- seed_oof[[1]]$patient
stopifnot(
  length(unique(reference_patients)) == length(reference_patients),
  all(vapply(seed_oof, function(x) setequal(x$patient, reference_patients), logical(1)))
)

make_scenario <- function(complete_case = FALSE) {
  first <- seed_oof[[1]]
  keep <- if (complete_case) first$complete_case else rep(TRUE, nrow(first))
  base <- first[keep, c("patient", "cancer_type", "time", "event", "fold")]
  base <- base[order(base$patient), ]
  risk_column <- if (complete_case) {
    "risk_score_complete_case_calibrated"
  } else {
    "risk_score_calibrated"
  }
  risks <- sapply(seed_oof, function(frame) {
    frame <- frame[if (complete_case) frame$complete_case else rep(TRUE, nrow(frame)), ]
    frame[[risk_column]][match(base$patient, frame$patient)]
  })
  stopifnot(!anyNA(risks), all(is.finite(risks)))
  list(base = base, risks = risks)
}

master <- read.csv(master_file, stringsAsFactors = FALSE, check.names = FALSE)
master$event <- as_flag(master$event)
availability_columns <- c("has_rna", "has_cnv", "has_rppa", "has_methylation", "has_he")
master[availability_columns] <- lapply(master[availability_columns], as_flag)
master_complete <- rowSums(as.matrix(master[availability_columns])) == 5

scenarios <- list(
  natural_availability = make_scenario(FALSE),
  complete_case = make_scenario(TRUE)
)
scenario_masks <- list(
  natural_availability = rep(TRUE, nrow(master)),
  complete_case = master_complete
)
old_summary <- read.delim(old_summary_file, stringsAsFactors = FALSE, check.names = FALSE)

prepare_fold_weights <- function(scenario_name, scenario, cancer) {
  weights <- vector("list", length(folds))
  names(weights) <- as.character(folds)
  for (fold in folds) {
    train <- master[
      scenario_masks[[scenario_name]] & master$fold != fold & master$cancer_type == cancer,
    ]
    test_index <- which(scenario$base$fold == fold & scenario$base$cancer_type == cancer)
    test <- scenario$base[test_index, ]
    if (!nrow(train) || !nrow(test)) {
      weights[[as.character(fold)]] <- list(global_index = test_index, horizons = NULL)
      next
    }
    fit <- estimate_censoring_distribution(train$time, train$event)
    censoring_survival <- predict_censoring_survival(fit, test$time)
    if (any(censoring_survival[test$event] <= 0)) {
      weights[[as.character(fold)]] <- list(global_index = test_index, horizons = NULL)
      next
    }
    horizon_weights <- lapply(horizons, function(horizon) {
      list(
        case_weight = ifelse(
          test$event & test$time <= horizon,
          1 / censoring_survival,
          0
        ),
        control = as.integer(test$time > horizon)
      )
    })
    weights[[as.character(fold)]] <- list(
      global_index = test_index,
      horizons = horizon_weights
    )
  }
  weights
}

estimate_auc <- function(scenario, fold_weights, sampled_index, horizon_index) {
  sampled_fold <- scenario$base$fold[sampled_index]
  seed_values <- vapply(seq_along(seeds), function(seed_index) {
    fold_values <- rep(NA_real_, length(folds))
    fold_sizes <- integer(length(folds))
    for (fold_index in seq_along(folds)) {
      fold <- folds[fold_index]
      selected <- which(sampled_fold == fold)
      fold_sizes[fold_index] <- length(selected)
      definition <- fold_weights[[as.character(fold)]]
      if (!length(selected) || is.null(definition$horizons)) next
      global_index <- sampled_index[selected]
      local_index <- match(global_index, definition$global_index)
      weight <- definition$horizons[[horizon_index]]
      fold_values[fold_index] <- weighted_auc_cpp(
        scenario$risks[global_index, seed_index],
        weight$case_weight[local_index],
        weight$control[local_index]
      )
    }
    valid <- is.finite(fold_values) & fold_sizes > 0
    if (!any(valid)) return(NA_real_)
    weighted.mean(fold_values[valid], fold_sizes[valid])
  }, numeric(1))
  mean(seed_values, na.rm = TRUE)
}

set.seed(bootstrap_seed)
summary_rows <- list()
replicate_rows <- list()
summary_index <- 1L
replicate_index <- 1L

for (scenario_name in names(scenarios)) {
  scenario <- scenarios[[scenario_name]]
  for (cancer in sort(unique(master$cancer_type))) {
    cancer_index <- which(scenario$base$cancer_type == cancer)
    fold_weights <- prepare_fold_weights(scenario_name, scenario, cancer)
    for (horizon_index in seq_along(horizons)) {
      horizon_name <- names(horizons)[horizon_index]
      old <- old_summary[
        old_summary$scenario == scenario_name &
          old_summary$cancer_type == cancer &
          old_summary$horizon == horizon_name,
      ]
      stopifnot(nrow(old) == 1)
      estimate <- estimate_auc(scenario, fold_weights, cancer_index, horizon_index)
      if (is.finite(old$auc_mean)) {
        if (abs(estimate - old$auc_mean) >= 1e-10) {
          stop(
            scenario_name, " ", cancer, " ", horizon_name,
            ": recomputed=", estimate, ", previous=", old$auc_mean
          )
        }
      }

      bootstrap_values <- numeric(0)
      if (old$status == "evaluable") {
        bootstrap_values <- vapply(seq_len(n_bootstrap), function(i) {
          sampled <- sample(cancer_index, length(cancer_index), replace = TRUE)
          estimate_auc(scenario, fold_weights, sampled, horizon_index)
        }, numeric(1))
        bootstrap_values <- bootstrap_values[is.finite(bootstrap_values)]
      }

      summary_rows[[summary_index]] <- data.frame(
        scenario = scenario_name,
        cancer_type = cancer,
        horizon = horizon_name,
        estimate = estimate,
        lower_95ci = if (length(bootstrap_values)) {
          unname(quantile(bootstrap_values, 0.025, type = 7))
        } else NA_real_,
        upper_95ci = if (length(bootstrap_values)) {
          unname(quantile(bootstrap_values, 0.975, type = 7))
        } else NA_real_,
        n_valid_bootstrap = length(bootstrap_values),
        n_patients = old$n_patients,
        n_events = old$n_events,
        cases = old$cases,
        controls = old$controls,
        evaluable_folds = old$evaluable_folds,
        status = old$status
      )
      summary_index <- summary_index + 1L

      if (length(bootstrap_values)) {
        replicate_rows[[replicate_index]] <- data.frame(
          scenario = scenario_name,
          cancer_type = cancer,
          horizon = horizon_name,
          replicate = seq_along(bootstrap_values),
          auc = bootstrap_values
        )
        replicate_index <- replicate_index + 1L
      }
    }
  }
}

summary_data <- do.call(rbind, summary_rows)
replicate_data <- do.call(rbind, replicate_rows)
stopifnot(all(summary_data$n_valid_bootstrap[summary_data$status == "evaluable"] >= 950))

write.table(
  summary_data,
  file.path(output_directory, "EDFig1ab_cancer_specific_time_dependent_AUC_95CI.tsv"),
  sep = "\t", row.names = FALSE, quote = FALSE, na = "NA"
)
write.table(
  replicate_data,
  file.path(output_directory, "EDFig1ab_cancer_specific_AUC_bootstrap_replicates.tsv"),
  sep = "\t", row.names = FALSE, quote = FALSE, na = "NA"
)
write.table(
  data.frame(
    n_bootstrap = n_bootstrap,
    bootstrap_seed = bootstrap_seed,
    sampling_unit = "patient",
    seed_handling = "same sampled patients for all five seeds; AUC averaged across seeds",
    model_retraining = FALSE
  ),
  file.path(output_directory, "EDFig1ab_bootstrap_metadata.tsv"),
  sep = "\t", row.names = FALSE, quote = FALSE
)

message("EDFIG1_CANCER_SPECIFIC_AUC_95CI_COMPLETE")
