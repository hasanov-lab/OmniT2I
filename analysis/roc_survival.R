setwd(commandArgs(trailingOnly = TRUE)[1])

## Required package


## 1. Define the final analysis inputs
master_file <- "Inputs/master_folds.csv"

oof_root <- "Inputs/oof"

auc_file <- "Analysis/auc/per_seed.tsv"

fold_auc_file <- "Analysis/auc/fold_metrics.tsv"

seeds <- c(101, 202, 303, 404, 505)
folds <- 0:4
year_days <- 365.25

horizons <- data.frame(
  horizon = factor(
    c("1 year", "3 years", "5 years"),
    levels = c("1 year", "3 years", "5 years")
  ),
  years = c(1, 3, 5),
  time_days = c(1, 3, 5) * year_days
)

availability_columns <- c(
  "has_rna",
  "has_cnv",
  "has_rppa",
  "has_methylation",
  "has_he"
)


## 2. Read the cohort definitions and final AUC summaries
master_data <- read.csv(
  master_file,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

master_data$event <- as.logical(master_data$event)
master_data[, availability_columns] <- lapply(
  master_data[, availability_columns],
  as.logical
)

complete_case <- rowSums(
  as.matrix(master_data[, availability_columns])
) == length(availability_columns)

auc_data <- read.delim(
  auc_file,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

fold_auc_data <- read.delim(
  fold_auc_file,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

scenario_definitions <- list(
  natural_availability = list(
    cohort = "Natural availability",
    master_mask = rep(TRUE, nrow(master_data)),
    risk_column = "risk_score_calibrated"
  ),
  complete_case = list(
    cohort = "Complete cases",
    master_mask = complete_case,
    risk_column = "risk_score_complete_case_calibrated"
  )
)


## 3. Estimate the outer-training censoring distribution
estimate_censoring_distribution <- function(time, event) {
  unique_times <- sort(unique(time))

  n_at_risk <- vapply(
    unique_times,
    function(time_value) sum(time >= time_value),
    numeric(1)
  )

  deaths <- vapply(
    unique_times,
    function(time_value) sum(time == time_value & event),
    numeric(1)
  )

  censoring_events <- vapply(
    unique_times,
    function(time_value) sum(time == time_value & !event),
    numeric(1)
  )

  censoring_risk_set <- n_at_risk - deaths
  censoring_hazard <- ifelse(
    censoring_events == 0,
    0,
    censoring_events / censoring_risk_set
  )

  list(
    time = unique_times,
    survival = cumprod(1 - censoring_hazard)
  )
}


predict_censoring_survival <- function(censoring_fit, time) {
  time_index <- findInterval(time, censoring_fit$time)
  survival <- rep(1, length(time))
  has_estimate <- time_index > 0

  survival[has_estimate] <- censoring_fit$survival[
    time_index[has_estimate]
  ]

  survival
}


## 4. Calculate one cumulative/dynamic IPCW ROC curve
calculate_ipcw_roc <- function(train_data, test_data, risk_column, horizon) {
  censoring_fit <- estimate_censoring_distribution(
    train_data$time,
    train_data$event
  )

  censoring_survival <- predict_censoring_survival(
    censoring_fit,
    test_data$time
  )

  if (any(censoring_survival[test_data$event] <= 0)) {
    stop("The outer-training censoring survival reaches zero.")
  }

  ipcw <- ifelse(
    test_data$event,
    1 / censoring_survival,
    0
  )

  is_case <- test_data$event & test_data$time <= horizon
  is_control <- test_data$time > horizon

  case_weight <- is_case * ipcw
  control_weight <- as.numeric(is_control)

  if (sum(case_weight) == 0 || sum(control_weight) == 0) {
    stop("A time horizon has no evaluable cases or controls.")
  }

  risk <- test_data[[risk_column]]

  if (any(!is.finite(risk))) {
    stop("Non-finite risk scores were found.")
  }

  risk_order <- order(-risk, seq_along(risk))
  ordered_risk <- risk[risk_order]
  ordered_case_weight <- case_weight[risk_order]
  ordered_control_weight <- control_weight[risk_order]

  true_positive_rate <- cumsum(ordered_case_weight) /
    sum(ordered_case_weight)
  false_positive_rate <- cumsum(ordered_control_weight) /
    sum(ordered_control_weight)

  keep_last_tied_score <- c(
    abs(diff(ordered_risk)) > 1e-8,
    TRUE
  )

  data.frame(
    false_positive_rate = c(
      0,
      false_positive_rate[keep_last_tied_score]
    ),
    true_positive_rate = c(
      0,
      true_positive_rate[keep_last_tied_score]
    )
  )
}


calculate_auc <- function(roc_curve) {
  sum(
    diff(roc_curve$false_positive_rate) *
      (
        head(roc_curve$true_positive_rate, -1) +
          tail(roc_curve$true_positive_rate, -1)
      ) / 2
  )
}


interpolate_roc <- function(roc_curve, false_positive_grid) {
  collapsed_curve <- aggregate(
    true_positive_rate ~ false_positive_rate,
    data = roc_curve,
    FUN = max
  )

  approx(
    x = collapsed_curve$false_positive_rate,
    y = collapsed_curve$true_positive_rate,
    xout = false_positive_grid,
    method = "linear",
    rule = 2
  )$y
}


## 5. Calculate fold-specific curves and verify their AUC values
false_positive_grid <- seq(0, 1, length.out = 401)
fold_curve_list <- list()
auc_check_list <- list()
list_index <- 1

for (scenario_name in names(scenario_definitions)) {
  scenario <- scenario_definitions[[scenario_name]]
  scenario_size <- sum(scenario$master_mask)

  for (seed_value in seeds) {
    for (fold_value in folds) {
      oof_file <- file.path(
        oof_root,
        paste0("seed_", seed_value),
        paste0("fold", fold_value),
        "survival_oof_predictions.tsv.gz"
      )

      oof_data <- read.delim(
        oof_file,
        stringsAsFactors = FALSE,
        check.names = FALSE
      )

      oof_data$event <- as.logical(oof_data$event)
      oof_data$complete_case <- as.logical(oof_data$complete_case)

      train_data <- master_data[
        scenario$master_mask & master_data$fold != fold_value,
      ]

      test_data <- oof_data

      if (scenario_name == "complete_case") {
        test_data <- test_data[test_data$complete_case, ]
      }

      fold_weight <- nrow(test_data) / scenario_size

      for (horizon_index in seq_len(nrow(horizons))) {
        horizon_name <- as.character(horizons$horizon[horizon_index])
        horizon_years <- horizons$years[horizon_index]
        horizon_days <- horizons$time_days[horizon_index]

        roc_curve <- calculate_ipcw_roc(
          train_data,
          test_data,
          scenario$risk_column,
          horizon_days
        )

        calculated_auc <- calculate_auc(roc_curve)
        saved_auc_column <- paste0("auc_", horizon_years, "y")
        saved_auc <- fold_auc_data[
          fold_auc_data$scenario == scenario_name &
            fold_auc_data$seed == seed_value &
            fold_auc_data$fold == fold_value,
          saved_auc_column
        ]

        auc_check_list[[list_index]] <- data.frame(
          scenario = scenario_name,
          seed = seed_value,
          fold = fold_value,
          horizon = horizon_name,
          calculated_auc = calculated_auc,
          saved_auc = saved_auc,
          absolute_difference = abs(calculated_auc - saved_auc)
        )

        fold_curve_list[[list_index]] <- data.frame(
          scenario = scenario_name,
          cohort = scenario$cohort,
          seed = seed_value,
          fold = fold_value,
          horizon = horizon_name,
          false_positive_rate = false_positive_grid,
          true_positive_rate = interpolate_roc(
            roc_curve,
            false_positive_grid
          ),
          fold_weight = fold_weight
        )

        list_index <- list_index + 1
      }
    }
  }
}

auc_check <- do.call(rbind, auc_check_list)

if (max(auc_check$absolute_difference) > 1e-10) {
  stop("The reconstructed ROC curves do not reproduce the saved AUC values.")
}

fold_curves <- do.call(rbind, fold_curve_list)


## 6. Average five folds within each seed, then average five seeds
fold_curves$weighted_true_positive_rate <-
  fold_curves$true_positive_rate * fold_curves$fold_weight

seed_curves <- aggregate(
  weighted_true_positive_rate ~
    cohort + seed + horizon + false_positive_rate,
  data = fold_curves,
  FUN = sum
)

names(seed_curves)[5] <- "true_positive_rate"

fig3b_roc_data <- aggregate(
  true_positive_rate ~ cohort + horizon + false_positive_rate,
  data = seed_curves,
  FUN = mean
)

fig3b_roc_data$horizon <- factor(
  fig3b_roc_data$horizon,
  levels = levels(horizons$horizon)
)

fig3b_roc_data$cohort <- factor(
  fig3b_roc_data$cohort,
  levels = c("Natural availability", "Complete cases")
)


## 7. Prepare the AUC labels
fig3b_auc_data <- rbind(
  data.frame(
    seed = auc_data$seed,
    scenario = auc_data$scenario,
    horizon = "1 year",
    auc = auc_data$auc_1y
  ),
  data.frame(
    seed = auc_data$seed,
    scenario = auc_data$scenario,
    horizon = "3 years",
    auc = auc_data$auc_3y
  ),
  data.frame(
    seed = auc_data$seed,
    scenario = auc_data$scenario,
    horizon = "5 years",
    auc = auc_data$auc_5y
  )
)

fig3b_auc_data$cohort <- ifelse(
  fig3b_auc_data$scenario == "natural_availability",
  "Natural availability",
  "Complete cases"
)

fig3b_auc_summary <- read.delim(
  "Source_Data/Figure3/Fig3b_time_dependent_AUC_95CI.tsv",
  stringsAsFactors = FALSE,
  check.names = FALSE
)

fig3b_auc_summary$horizon <- factor(
  fig3b_auc_summary$horizon,
  levels = levels(horizons$horizon)
)

fig3b_auc_summary$cohort <- factor(
  fig3b_auc_summary$cohort,
  levels = c("Natural availability", "Complete cases")
)

fig3b_auc_summary$label <- sprintf(
  "%s: %.3f (%.3f-%.3f)",
  fig3b_auc_summary$cohort,
  fig3b_auc_summary$estimate,
  fig3b_auc_summary$lower_95ci,
  fig3b_auc_summary$upper_95ci
)

fig3b_auc_summary$label_y <- ifelse(
  fig3b_auc_summary$cohort == "Natural availability",
  0.15,
  0.07
)


## 8. Export source data
write.table(
  fig3b_auc_data,
  "Source_Data/Figure3/Fig3b_time_dependent_AUC_source_data.tsv",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

write.table(
  fig3b_roc_data,
  "Source_Data/Figure3/Fig3b_time_dependent_ROC_source_data.tsv",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

write.table(
  auc_check,
  "Source_Data/Figure3/Fig3b_ROC_AUC_validation.tsv",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)
