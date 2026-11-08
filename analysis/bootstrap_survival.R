setwd(commandArgs(trailingOnly = TRUE)[1])

library(Rcpp)
started <- Sys.time()

## Patient-level bootstrap confidence intervals for Figure 3.
## Existing outer-test predictions are reused; no model is retrained.

seeds <- c(101, 202, 303, 404, 505)
folds <- 0:4
n_bootstrap <- 1000L
bootstrap_seed <- 20260901L
year_days <- 365.25
horizons <- c(`1 year` = year_days, `3 years` = 3 * year_days,
              `5 years` = 5 * year_days)

oof_root <- "Inputs/oof"

master_file <- "Inputs/master_folds.csv"

source_directory <- "Source_Data/Figure3"
dir.create(source_directory, recursive = TRUE, showWarnings = FALSE)

Rcpp::cppFunction('
double harrell_c_index_cpp(
    NumericVector time,
    LogicalVector event,
    NumericVector risk,
    IntegerVector stratum) {
  int n = time.size();
  if (event.size() != n || risk.size() != n || stratum.size() != n) {
    stop("Harrell C-index inputs have different lengths");
  }

  std::map<int, std::vector<int> > groups;
  for (int i = 0; i < n; ++i) groups[stratum[i]].push_back(i);

  double concordant = 0.0;
  double tied = 0.0;
  double comparable = 0.0;

  for (std::map<int, std::vector<int> >::iterator group = groups.begin();
       group != groups.end(); ++group) {
    std::vector<int> idx = group->second;
    if (idx.size() < 2) continue;

    std::vector<double> risk_values;
    risk_values.reserve(idx.size());
    for (size_t j = 0; j < idx.size(); ++j) risk_values.push_back(risk[idx[j]]);
    std::sort(risk_values.begin(), risk_values.end());
    risk_values.erase(std::unique(risk_values.begin(), risk_values.end()),
                      risk_values.end());

    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
      if (time[a] != time[b]) return time[a] > time[b];
      return a < b;
    });

    std::vector<double> bit(risk_values.size() + 1, 0.0);
    auto bit_add = [&](int position, double value) {
      for (int p = position; p < (int)bit.size(); p += p & -p) bit[p] += value;
    };
    auto bit_sum = [&](int position) {
      double total = 0.0;
      for (int p = position; p > 0; p -= p & -p) total += bit[p];
      return total;
    };

    size_t start = 0;
    while (start < idx.size()) {
      size_t end = start + 1;
      while (end < idx.size() && time[idx[end]] == time[idx[start]]) ++end;

      double later_total = bit_sum((int)risk_values.size());
      for (size_t j = start; j < end; ++j) {
        int row = idx[j];
        if (!event[row]) continue;
        int rank = std::lower_bound(risk_values.begin(), risk_values.end(),
                                    risk[row]) - risk_values.begin() + 1;
        double lower = bit_sum(rank - 1);
        double equal = bit_sum(rank) - lower;

        // scikit-survival also compares an event with censoring at the same time.
        for (size_t k = start; k < end; ++k) {
          int tied_row = idx[k];
          if (event[tied_row]) continue;
          if (risk[tied_row] < risk[row]) lower += 1.0;
          else if (risk[tied_row] == risk[row]) equal += 1.0;
        }

        concordant += lower;
        tied += equal;
        comparable += later_total;
        for (size_t k = start; k < end; ++k) {
          if (!event[idx[k]]) comparable += 1.0;
        }
      }

      for (size_t j = start; j < end; ++j) {
        int row = idx[j];
        int rank = std::lower_bound(risk_values.begin(), risk_values.end(),
                                    risk[row]) - risk_values.begin() + 1;
        bit_add(rank, 1.0);
      }
      start = end;
    }
  }

  if (comparable == 0.0) return NA_REAL;
  return (concordant + 0.5 * tied) / comparable;
}
')

Rcpp::cppFunction('
double weighted_auc_cpp(
    NumericVector risk,
    NumericVector case_weight,
    IntegerVector control) {
  int n = risk.size();
  if (case_weight.size() != n || control.size() != n) {
    stop("AUC inputs have different lengths");
  }

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
    double n_lower = lower - control_risk.begin();
    double n_equal = upper - lower;
    numerator += case_weight[i] * (n_lower + 0.5 * n_equal);
  }

  return numerator / (total_case_weight * control_risk.size());
}
')

read_seed_oof <- function(seed_value) {
  frames <- lapply(folds, function(fold_value) {
    path <- file.path(
      oof_root,
      paste0("seed_", seed_value),
      paste0("fold", fold_value),
      "survival_oof_predictions.tsv.gz"
    )
    frame <- read.delim(path, stringsAsFactors = FALSE, check.names = FALSE)
    frame$event <- as.logical(frame$event)
    frame$complete_case <- as.logical(frame$complete_case)
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
  colnames(risks) <- as.character(seeds)
  list(base = base, risks = risks, risk_column = risk_column)
}

scenarios <- list(
  natural_availability = make_scenario(FALSE),
  complete_case = make_scenario(TRUE)
)

sample_by_cancer <- function(cancer_type) {
  groups <- split(seq_along(cancer_type), cancer_type)
  unlist(lapply(groups, function(index) sample(index, length(index), replace = TRUE)),
         use.names = FALSE)
}

seed_c_indices <- function(scenario, sampled_index) {
  data <- scenario$base[sampled_index, ]
  risks <- scenario$risks[sampled_index, , drop = FALSE]
  pooled_stratum <- rep.int(1L, nrow(data))
  cancer_stratum <- as.integer(factor(data$cancer_type))

  pooled <- vapply(seq_along(seeds), function(seed_index) {
    harrell_c_index_cpp(data$time, data$event, risks[, seed_index], pooled_stratum)
  }, numeric(1))
  within <- vapply(seq_along(seeds), function(seed_index) {
    harrell_c_index_cpp(data$time, data$event, risks[, seed_index], cancer_stratum)
  }, numeric(1))
  c(`All patients` = mean(pooled), `Within cancer` = mean(within))
}

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

master_data <- read.csv(master_file, stringsAsFactors = FALSE, check.names = FALSE)
master_data$event <- as.logical(master_data$event)
availability_columns <- c("has_rna", "has_cnv", "has_rppa", "has_methylation", "has_he")
master_data[, availability_columns] <- lapply(master_data[, availability_columns], as.logical)
master_complete <- rowSums(as.matrix(master_data[, availability_columns])) == 5

prepare_auc_weights <- function(scenario_name, scenario) {
  scenario_mask <- if (scenario_name == "complete_case") master_complete else rep(TRUE, nrow(master_data))
  weights <- vector("list", length(folds))
  names(weights) <- as.character(folds)

  for (fold_value in folds) {
    train <- master_data[scenario_mask & master_data$fold != fold_value, ]
    test_index <- which(scenario$base$fold == fold_value)
    test <- scenario$base[test_index, ]
    fit <- estimate_censoring_distribution(train$time, train$event)
    censoring_survival <- predict_censoring_survival(fit, test$time)
    if (any(censoring_survival[test$event] <= 0)) {
      stop("Outer-training censoring survival reaches zero")
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
    weights[[as.character(fold_value)]] <- list(
      global_index = test_index,
      horizons = horizon_weights
    )
  }
  weights
}

auc_weights <- Map(prepare_auc_weights, names(scenarios), scenarios)

seed_auc_values <- function(scenario_name, scenario, sampled_index) {
  sampled_fold <- scenario$base$fold[sampled_index]
  result <- matrix(NA_real_, nrow = length(seeds), ncol = length(horizons),
                   dimnames = list(as.character(seeds), names(horizons)))

  for (seed_index in seq_along(seeds)) {
    fold_values <- matrix(NA_real_, nrow = length(folds), ncol = length(horizons))
    fold_sizes <- integer(length(folds))

    for (fold_index in seq_along(folds)) {
      fold_value <- folds[fold_index]
      selected <- which(sampled_fold == fold_value)
      fold_sizes[fold_index] <- length(selected)
      if (!length(selected)) next

      original_global <- sampled_index[selected]
      fold_definition <- auc_weights[[scenario_name]][[as.character(fold_value)]]
      local_index <- match(original_global, fold_definition$global_index)
      fold_risk <- scenario$risks[original_global, seed_index]

      for (horizon_index in seq_along(horizons)) {
        weight <- fold_definition$horizons[[horizon_index]]
        fold_values[fold_index, horizon_index] <- weighted_auc_cpp(
          fold_risk,
          weight$case_weight[local_index],
          weight$control[local_index]
        )
      }
    }

    for (horizon_index in seq_along(horizons)) {
      valid <- is.finite(fold_values[, horizon_index]) & fold_sizes > 0
      result[seed_index, horizon_index] <- weighted.mean(
        fold_values[valid, horizon_index],
        fold_sizes[valid]
      )
    }
  }
  result
}

set.seed(bootstrap_seed)

fig3a_replicates <- list()
fig3b_replicates <- list()
fig3ef_replicates <- list()

for (scenario_name in names(scenarios)) {
  scenario <- scenarios[[scenario_name]]
  cohort_label <- if (scenario_name == "natural_availability") {
    "Natural availability"
  } else {
    "Complete cases"
  }

  original_index <- seq_len(nrow(scenario$base))
  original_c <- seed_c_indices(scenario, original_index)
  original_auc <- seed_auc_values(scenario_name, scenario, original_index)

  c_replicates <- matrix(NA_real_, nrow = n_bootstrap, ncol = 2,
                         dimnames = list(NULL, names(original_c)))
  auc_replicates <- matrix(NA_real_, nrow = n_bootstrap, ncol = length(horizons),
                           dimnames = list(NULL, names(horizons)))

  for (replicate_index in seq_len(n_bootstrap)) {
    sampled <- sample_by_cancer(scenario$base$cancer_type)
    c_replicates[replicate_index, ] <- seed_c_indices(scenario, sampled)
    auc_replicates[replicate_index, ] <- colMeans(
      seed_auc_values(scenario_name, scenario, sampled)
    )
  }

  fig3a_replicates[[scenario_name]] <- data.frame(
    replicate = seq_len(n_bootstrap),
    cohort = cohort_label,
    `All patients` = c_replicates[, "All patients"],
    `Within cancer` = c_replicates[, "Within cancer"],
    check.names = FALSE
  )

  fig3b_replicates[[scenario_name]] <- data.frame(
    replicate = seq_len(n_bootstrap),
    cohort = cohort_label,
    auc_replicates,
    check.names = FALSE
  )

  for (cancer_name in sort(unique(scenario$base$cancer_type))) {
    cancer_index <- which(scenario$base$cancer_type == cancer_name)
    cancer_data <- scenario$base[cancer_index, ]
    cancer_risks <- scenario$risks[cancer_index, , drop = FALSE]
    cancer_original <- vapply(seq_along(seeds), function(seed_index) {
      harrell_c_index_cpp(
        cancer_data$time,
        cancer_data$event,
        cancer_risks[, seed_index],
        rep.int(1L, nrow(cancer_data))
      )
    }, numeric(1))

    cancer_bootstrap <- vapply(seq_len(n_bootstrap), function(replicate_index) {
      sampled <- sample(seq_len(nrow(cancer_data)), nrow(cancer_data), replace = TRUE)
      values <- vapply(seq_along(seeds), function(seed_index) {
        harrell_c_index_cpp(
          cancer_data$time[sampled],
          cancer_data$event[sampled],
          cancer_risks[sampled, seed_index],
          rep.int(1L, length(sampled))
        )
      }, numeric(1))
      mean(values, na.rm = TRUE)
    }, numeric(1))

    fig3ef_replicates[[paste(scenario_name, cancer_name, sep = "|")]] <- data.frame(
      replicate = seq_len(n_bootstrap),
      scenario = scenario_name,
      cohort = cohort_label,
      cancer_type = cancer_name,
      n_patients = nrow(cancer_data),
      n_events = sum(cancer_data$event),
      estimate = mean(cancer_original, na.rm = TRUE),
      value = cancer_bootstrap
    )
  }
}

fig3a_replicates <- do.call(rbind, fig3a_replicates)
fig3b_replicates <- do.call(rbind, fig3b_replicates)
fig3ef_replicates <- do.call(rbind, fig3ef_replicates)

reshape_replicates <- function(data, columns, name_column, value_column) {
  output <- lapply(columns, function(column) {
    data.frame(
      replicate = data$replicate,
      cohort = data$cohort,
      name = column,
      value = data[[column]],
      stringsAsFactors = FALSE
    )
  })
  output <- do.call(rbind, output)
  names(output)[names(output) == "name"] <- name_column
  names(output)[names(output) == "value"] <- value_column
  output
}

fig3a_long <- reshape_replicates(
  fig3a_replicates,
  c("All patients", "Within cancer"),
  "metric",
  "c_index"
)

fig3b_long <- reshape_replicates(
  fig3b_replicates,
  names(horizons),
  "horizon",
  "auc"
)

summarize_ci <- function(data, group_columns, value_column) {
  groups <- interaction(data[, group_columns, drop = FALSE], drop = TRUE)
  pieces <- split(data, groups)
  output <- lapply(pieces, function(frame) {
    values <- frame[[value_column]]
    values <- values[is.finite(values)]
    row <- frame[1, group_columns, drop = FALSE]
    row$lower_95ci <- unname(quantile(values, 0.025, type = 7))
    row$upper_95ci <- unname(quantile(values, 0.975, type = 7))
    row$n_valid_bootstrap <- length(values)
    row
  })
  do.call(rbind, output)
}

fig3a_ci <- summarize_ci(fig3a_long, c("cohort", "metric"), "c_index")
harrell_seed <- read.delim(
  "Inputs/performance_per_seed.tsv",
  stringsAsFactors = FALSE,
  check.names = FALSE
)
harrell_seed$cohort <- ifelse(
  harrell_seed$scenario == "natural_availability",
  "Natural availability",
  "Complete cases"
)
harrell_long <- rbind(
  data.frame(cohort = harrell_seed$cohort, metric = "All patients",
             value = harrell_seed$pooled_c_index),
  data.frame(cohort = harrell_seed$cohort, metric = "Within cancer",
             value = harrell_seed$within_cancer_c_index)
)
harrell_summary <- aggregate(value ~ cohort + metric, harrell_long,
                             function(x) c(mean = mean(x), sd = sd(x)))
fig3a_ci$estimate <- harrell_summary$value[
  match(
    paste(fig3a_ci$cohort, fig3a_ci$metric),
    paste(harrell_summary$cohort, harrell_summary$metric)
  ), "mean"
]
fig3a_ci$seed_sd <- harrell_summary$value[
  match(
    paste(fig3a_ci$cohort, fig3a_ci$metric),
    paste(harrell_summary$cohort, harrell_summary$metric)
  ), "sd"
]

fig3b_ci <- summarize_ci(fig3b_long, c("cohort", "horizon"), "auc")
auc_seed <- read.delim(
  "Analysis/auc/per_seed.tsv",
  stringsAsFactors = FALSE,
  check.names = FALSE
)
auc_seed$cohort <- ifelse(
  auc_seed$scenario == "natural_availability",
  "Natural availability",
  "Complete cases"
)
auc_long <- rbind(
  data.frame(cohort = auc_seed$cohort, horizon = "1 year", value = auc_seed$auc_1y),
  data.frame(cohort = auc_seed$cohort, horizon = "3 years", value = auc_seed$auc_3y),
  data.frame(cohort = auc_seed$cohort, horizon = "5 years", value = auc_seed$auc_5y)
)
auc_summary <- aggregate(value ~ cohort + horizon, auc_long,
                         function(x) c(mean = mean(x), sd = sd(x)))
fig3b_ci$estimate <- auc_summary$value[
  match(
    paste(fig3b_ci$cohort, fig3b_ci$horizon),
    paste(auc_summary$cohort, auc_summary$horizon)
  ), "mean"
]
fig3b_ci$seed_sd <- auc_summary$value[
  match(
    paste(fig3b_ci$cohort, fig3b_ci$horizon),
    paste(auc_summary$cohort, auc_summary$horizon)
  ), "sd"
]

fig3ef_ci <- summarize_ci(
  fig3ef_replicates,
  c("scenario", "cohort", "cancer_type", "n_patients", "n_events", "estimate"),
  "value"
)


## 2. Derive current per-cancer point estimates; never join old-model estimates
current_rows <- lapply(names(scenarios), function(scenario_name) {
  scenario <- scenarios[[scenario_name]]
  do.call(rbind, lapply(sort(unique(master_data$cancer_type)), function(cancer) {
    selected <- which(scenario$base$cancer_type == cancer)
    d <- scenario$base[selected, ]
    values <- vapply(seq_along(seeds), function(j) {
      harrell_c_index_cpp(d$time, d$event, scenario$risks[selected, j], rep.int(1L, nrow(d)))
    }, numeric(1))
    data.frame(scenario = scenario_name, cancer_type = cancer, n_patients = nrow(d),
      n_events = sum(d$event), c_index_mean = if (any(is.finite(values))) mean(values, na.rm = TRUE) else NA_real_,
      c_index_sd = sd(values, na.rm = TRUE), n_evaluable_seeds = sum(is.finite(values)),
      estimability = if (any(is.finite(values))) 'estimable' else 'not evaluable',
      scenario_label = if (scenario_name == 'natural_availability') 'Natural availability' else 'Complete cases')
  }))
})
current <- do.call(rbind, current_rows)
ci_key <- paste(fig3ef_ci$scenario, fig3ef_ci$cancer_type)
index <- match(paste(current$scenario, current$cancer_type), ci_key)
current$lower_95ci <- fig3ef_ci$lower_95ci[index]
current$upper_95ci <- fig3ef_ci$upper_95ci[index]
current$n_valid_bootstrap <- fig3ef_ci$n_valid_bootstrap[index]
current$estimate <- current$c_index_mean
current$cohort <- current$scenario_label
stopifnot(max(abs(current$c_index_mean - fig3ef_ci$estimate[index]), na.rm = TRUE) < 1e-10,
          nrow(current) == 64L, all(fig3a_ci$n_valid_bootstrap == n_bootstrap),
          all(fig3b_ci$n_valid_bootstrap == n_bootstrap))

## 3. Verify updated point estimates against the already audited summaries
for (scenario_name in names(scenarios)) {
  cohort <- if (scenario_name == 'natural_availability') 'Natural availability' else 'Complete cases'
  calculated <- seed_c_indices(scenarios[[scenario_name]], seq_len(nrow(scenarios[[scenario_name]]$base)))
  saved <- fig3a_ci[fig3a_ci$cohort == cohort, ]
  stopifnot(max(abs(calculated[as.character(saved$metric)] - saved$estimate)) < 1e-7)
}

## 4. Save current-model tables
tables <- list(Fig3a_C_index_95CI = fig3a_ci, Fig3a_bootstrap_replicates = fig3a_long,
  Fig3b_time_dependent_AUC_95CI = fig3b_ci, Fig3b_bootstrap_replicates = fig3b_long,
  Fig3ef_cancer_specific_95CI = current, Fig3ef_cancer_bootstrap_replicates = fig3ef_replicates)
for (name in names(tables)) write.table(tables[[name]], file.path(source_directory, paste0(name, '.tsv')),
  sep = '\t', quote = FALSE, row.names = FALSE)
write.table(data.frame(n_bootstrap = n_bootstrap, bootstrap_seed = bootstrap_seed,
  sampling_unit = 'patient', sampling_scheme = 'within-cancer sampling with replacement',
  seed_handling = 'same patients across five seeds; mean metric',
  model_retraining = FALSE, elapsed_seconds = as.numeric(difftime(Sys.time(), started, units = 'secs'))),
  file.path(source_directory, 'bootstrap_metadata.tsv'), sep = '\t', quote = FALSE, row.names = FALSE)
message('CURRENT_FIGURE3_SOURCE_DATA_COMPLETE')

file.copy(file.path(source_directory, "Fig3ef_cancer_specific_95CI.tsv"), file.path(source_directory, "Fig3ef_cancer_specific_95CI_eligible.tsv"), overwrite = TRUE)
