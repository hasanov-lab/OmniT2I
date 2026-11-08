setwd(commandArgs(trailingOnly = TRUE)[1])

master_file <- "Inputs/master_folds.csv"
oof_root <- "Inputs/oof"
out <- "Source_Data/ExtendedData1"
seeds <- c(101, 202, 303, 404, 505)
folds <- 0:4
horizons <- c(`1 year` = 365.25, `3 years` = 1095.75, `5 years` = 1826.25)
availability_columns <- c("has_rna", "has_cnv", "has_rppa", "has_methylation", "has_he")

as_flag <- function(x) tolower(as.character(x)) == "true"

estimate_censoring_distribution <- function(time, event) {
  unique_times <- sort(unique(time))
  n_at_risk <- vapply(unique_times, function(t) sum(time >= t), numeric(1))
  deaths <- vapply(unique_times, function(t) sum(time == t & event), numeric(1))
  censored <- vapply(unique_times, function(t) sum(time == t & !event), numeric(1))
  risk_set <- n_at_risk - deaths
  hazard <- ifelse(censored == 0, 0, censored / risk_set)
  list(time = unique_times, survival = cumprod(1 - hazard))
}

predict_censoring_survival <- function(fit, time) {
  index <- findInterval(time, fit$time)
  survival <- rep(1, length(time))
  survival[index > 0] <- fit$survival[index[index > 0]]
  survival
}

calculate_auc <- function(train, test, risk_column, horizon) {
  if (!nrow(train) || !nrow(test)) return(list(auc = NA_real_, reason = "no test or training patients"))
  fit <- estimate_censoring_distribution(train$time, train$event)
  censoring_survival <- predict_censoring_survival(fit, test$time)
  if (any(censoring_survival[test$event] <= 0)) return(list(auc = NA_real_, reason = "insufficient censoring support"))

  is_case <- test$event & test$time <= horizon
  is_control <- test$time > horizon
  case_weight <- is_case * ifelse(test$event, 1 / censoring_survival, 0)
  control_weight <- as.numeric(is_control)
  if (sum(case_weight) == 0) return(list(auc = NA_real_, reason = "no evaluable cases"))
  if (sum(control_weight) == 0) return(list(auc = NA_real_, reason = "no evaluable controls"))

  risk <- test[[risk_column]]
  if (any(!is.finite(risk))) return(list(auc = NA_real_, reason = "non-finite risk"))
  risk_order <- order(-risk, seq_along(risk))
  ordered_risk <- risk[risk_order]
  true_positive_rate <- cumsum(case_weight[risk_order]) / sum(case_weight)
  false_positive_rate <- cumsum(control_weight[risk_order]) / sum(control_weight)
  keep <- c(abs(diff(ordered_risk)) > 1e-8, TRUE)
  x <- c(0, false_positive_rate[keep])
  y <- c(0, true_positive_rate[keep])
  list(auc = sum(diff(x) * (head(y, -1) + tail(y, -1)) / 2), reason = "evaluable")
}

dir.create(out, recursive = TRUE, showWarnings = FALSE)
master <- read.csv(master_file, stringsAsFactors = FALSE, check.names = FALSE)
master$event <- as_flag(master$event)
master[availability_columns] <- lapply(master[availability_columns], as_flag)
complete_case <- rowSums(as.matrix(master[availability_columns])) == length(availability_columns)
scenarios <- list(
  natural_availability = list(mask = rep(TRUE, nrow(master)), risk = "risk_score_calibrated"),
  complete_case = list(mask = complete_case, risk = "risk_score_complete_case_calibrated")
)

rows <- list()
row_index <- 1
for (seed in seeds) {
  for (fold in folds) {
    oof <- read.delim(
      file.path(oof_root, paste0("seed_", seed), paste0("fold", fold), "survival_oof_predictions.tsv.gz"),
      stringsAsFactors = FALSE, check.names = FALSE
    )
    oof$event <- as_flag(oof$event)
    oof$complete_case <- as_flag(oof$complete_case)

    for (scenario_name in names(scenarios)) {
      scenario <- scenarios[[scenario_name]]
      scenario_master <- master[scenario$mask, ]
      scenario_test <- if (scenario_name == "complete_case") oof[oof$complete_case, ] else oof

      for (cancer in sort(unique(master$cancer_type))) {
        train <- scenario_master[scenario_master$fold != fold & scenario_master$cancer_type == cancer, ]
        test <- scenario_test[scenario_test$cancer_type == cancer, ]
        for (horizon_name in names(horizons)) {
          result <- calculate_auc(train, test, scenario$risk, horizons[[horizon_name]])
          rows[[row_index]] <- data.frame(
            scenario = scenario_name, cancer_type = cancer, seed = seed, fold = fold,
            horizon = horizon_name, horizon_days = horizons[[horizon_name]],
            n_train = nrow(train), n_test = nrow(test),
            train_events = sum(train$event), test_events = sum(test$event),
            cases = sum(test$event & test$time <= horizons[[horizon_name]]),
            controls = sum(test$time > horizons[[horizon_name]]),
            auc = result$auc, status = result$reason
          )
          row_index <- row_index + 1
        }
      }
    }
  }
}

per_fold <- do.call(rbind, rows)
seed_groups <- split(per_fold, interaction(per_fold$scenario, per_fold$cancer_type, per_fold$seed, per_fold$horizon, drop = TRUE))
per_seed <- do.call(rbind, lapply(seed_groups, function(d) {
  ok <- is.finite(d$auc)
  data.frame(
    scenario = d$scenario[1], cancer_type = d$cancer_type[1], seed = d$seed[1], horizon = d$horizon[1],
    auc = if (any(ok)) weighted.mean(d$auc[ok], d$n_test[ok]) else NA_real_,
    n_patients = sum(d$n_test), n_events = sum(d$test_events),
    cases = sum(d$cases), controls = sum(d$controls),
    evaluable_folds = sum(ok), evaluable_patients = sum(d$n_test[ok]),
    status = if (any(ok)) "evaluable" else paste(unique(d$status), collapse = "; ")
  )
}))
row.names(per_seed) <- NULL

summary_groups <- split(per_seed, interaction(per_seed$scenario, per_seed$cancer_type, per_seed$horizon, drop = TRUE))
summary_data <- do.call(rbind, lapply(summary_groups, function(d) {
  ok <- is.finite(d$auc)
  evaluable_folds <- if (any(ok)) min(d$evaluable_folds[ok]) else 0
  status <- if (!any(ok)) {
    paste(unique(d$status), collapse = "; ")
  } else if (d$cases[1] < 10) {
    "fewer than 10 cases"
  } else if (d$controls[1] < 10) {
    "fewer than 10 controls"
  } else if (evaluable_folds < 3) {
    "fewer than 3 evaluable folds"
  } else {
    "evaluable"
  }
  data.frame(
    scenario = d$scenario[1], cancer_type = d$cancer_type[1], horizon = d$horizon[1],
    auc_mean = if (any(ok)) mean(d$auc[ok]) else NA_real_,
    auc_sd = if (sum(ok) > 1) sd(d$auc[ok]) else NA_real_,
    n_seeds = sum(ok), n_patients = d$n_patients[1], n_events = d$n_events[1],
    cases = d$cases[1], controls = d$controls[1], evaluable_folds = evaluable_folds,
    status = status
  )
}))
row.names(summary_data) <- NULL

dir.create(out, recursive = TRUE, showWarnings = FALSE)
write.table(per_fold, file.path(out, "EDFig2cd_cancer_specific_td_auc_per_fold.tsv"), sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")
write.table(per_seed, file.path(out, "EDFig2cd_cancer_specific_td_auc_per_seed.tsv"), sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")
write.table(summary_data, file.path(out, "EDFig2cd_cancer_specific_td_auc_summary.tsv"), sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")
message("Wrote cancer-specific IPCW time-dependent AUC source data to ", out)
