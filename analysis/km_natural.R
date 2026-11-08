setwd(commandArgs(trailingOnly = TRUE)[1])


## Required packages
library(survival)


dir.create("Source_Data/Figure3", recursive = TRUE, showWarnings = FALSE)

## 1. Read and combine existing out-of-fold predictions
oof_root <- "Inputs/oof"

seeds <- c(101, 202, 303, 404, 505)
folds <- 0:4

oof_list <- list()
list_index <- 1

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

    oof_data$event <- oof_data$event == "True"

    oof_list[[list_index]] <- oof_data[, c(
      "patient",
      "cancer_type",
      "time",
      "event",
      "seed",
      "risk_score_calibrated"
    )]

    list_index <- list_index + 1
  }
}

all_oof <- do.call(rbind, oof_list)


## 2. Rank risk within each cancer and seed, then average five seeds
all_oof$within_cancer_percentile <- ave(
  all_oof$risk_score_calibrated,
  all_oof$seed,
  all_oof$cancer_type,
  FUN = function(x) {
    (rank(x, ties.method = "average") - 0.5) / length(x)
  }
)

risk_mean <- aggregate(
  within_cancer_percentile ~ patient + cancer_type + time + event,
  data = all_oof,
  FUN = mean
)

names(risk_mean)[5] <- "mean_within_cancer_percentile"

risk_mean$risk_group <- ave(
  risk_mean$mean_within_cancer_percentile,
  risk_mean$cancer_type,
  FUN = function(x) {
    ifelse(x > median(x), "Higher risk", "Lower risk")
  }
)

risk_mean$risk_group <- factor(
  risk_mean$risk_group,
  levels = c("Lower risk", "Higher risk")
)


## 3. Calculate Kaplan-Meier curves without fitting a new prediction model
calculate_km <- function(patient_data) {
  output_list <- list()

  for (group_name in levels(patient_data$risk_group)) {
    group_data <- patient_data[
      patient_data$risk_group == group_name,
    ]

    event_times <- sort(unique(group_data$time[group_data$event]))
    survival_probability <- 1
    greenwood_sum <- 0

    curve_data <- data.frame(
      time_years = 0,
      survival = 1,
      lower = 1,
      upper = 1,
      risk_group = group_name
    )

    for (event_time in event_times) {
      number_at_risk <- sum(group_data$time >= event_time)
      number_of_events <- sum(
        group_data$time == event_time & group_data$event
      )

      survival_probability <- survival_probability *
        (1 - number_of_events / number_at_risk)

      if (number_at_risk > number_of_events) {
        greenwood_sum <- greenwood_sum +
          number_of_events /
          (number_at_risk * (number_at_risk - number_of_events))
      }

      if (survival_probability > 0 && survival_probability < 1) {
        log_log_survival <- log(-log(survival_probability))
        log_log_standard_error <-
          sqrt(greenwood_sum) / abs(log(survival_probability))

        lower_confidence <- exp(
          -exp(log_log_survival + 1.96 * log_log_standard_error)
        )
        upper_confidence <- exp(
          -exp(log_log_survival - 1.96 * log_log_standard_error)
        )
      } else {
        lower_confidence <- survival_probability
        upper_confidence <- survival_probability
      }

      curve_data <- rbind(
        curve_data,
        data.frame(
          time_years = event_time / 365.25,
          survival = survival_probability,
          lower = lower_confidence,
          upper = upper_confidence,
          risk_group = group_name
        )
      )
    }

    output_list[[group_name]] <- curve_data
  }

  do.call(rbind, output_list)
}


calculate_censor_marks <- function(patient_data, curve_data) {
  censor_list <- list()

  for (group_name in levels(patient_data$risk_group)) {
    group_censors <- patient_data[
      patient_data$risk_group == group_name & !patient_data$event,
    ]
    group_curve <- curve_data[curve_data$risk_group == group_name,]
    censor_times <- group_censors$time / 365.25
    curve_index <- findInterval(censor_times, group_curve$time_years)

    censor_list[[group_name]] <- data.frame(
      time_years = censor_times,
      survival = group_curve$survival[pmax(curve_index, 1)],
      risk_group = group_name
    )
  }

  do.call(rbind, censor_list)
}


## 4. Calculate stratified survival statistics
calculate_stratified_logrank <- function(patient_data) {
  observed_minus_expected <- 0
  total_variance <- 0

  for (cancer_name in unique(patient_data$cancer_type)) {
    cancer_data <- patient_data[
      patient_data$cancer_type == cancer_name,
    ]

    event_times <- sort(unique(cancer_data$time[cancer_data$event]))

    for (event_time in event_times) {
      at_risk <- cancer_data$time >= event_time
      event_at_time <- cancer_data$time == event_time & cancer_data$event

      n_high <- as.numeric(sum(
        at_risk & cancer_data$risk_group == "Higher risk"
      ))
      n_low <- as.numeric(sum(
        at_risk & cancer_data$risk_group == "Lower risk"
      ))
      d_high <- as.numeric(sum(
        event_at_time & cancer_data$risk_group == "Higher risk"
      ))
      d_total <- as.numeric(sum(event_at_time))
      n_total <- n_high + n_low

      if (n_total > 1 && d_total > 0) {
        expected_high <- d_total * n_high / n_total
        variance_high <- n_high * n_low * d_total *
          (n_total - d_total) /
          (n_total^2 * (n_total - 1))

        observed_minus_expected <- observed_minus_expected +
          d_high - expected_high
        total_variance <- total_variance + variance_high
      }
    }
  }

  chi_square <- observed_minus_expected^2 / total_variance
  data.frame(
    test = "Cancer-stratified log-rank",
    chi_square = chi_square,
    degrees_of_freedom = 1,
    p_value = pchisq(chi_square, df = 1, lower.tail = FALSE)
  )
}


km_curve <- calculate_km(risk_mean)
censor_marks <- calculate_censor_marks(risk_mean, km_curve)
logrank_statistics <- calculate_stratified_logrank(risk_mean)
logrank_p <- logrank_statistics$p_value

cox_model <- coxph(
  Surv(time, event) ~ risk_group + strata(cancer_type),
  data = risk_mean,
  ties = "efron"
)

cox_summary <- summary(cox_model)
cox_ph_test <- cox.zph(cox_model)

hazard_ratio <- unname(cox_summary$conf.int[1, "exp(coef)"])
hazard_ratio_lower <- unname(cox_summary$conf.int[1, "lower .95"])
hazard_ratio_upper <- unname(cox_summary$conf.int[1, "upper .95"])
cox_wald_p <- unname(cox_summary$coefficients[1, "Pr(>|z|)"])
cox_ph_p <- unname(cox_ph_test$table[1, "p"])

if (logrank_p < 0.001) {
  p_label <- "p < 0.001"
} else {
  p_label <- paste0(
    "p = ",
    format(logrank_p, digits = 2)
  )
}

hr_label <- sprintf(
  "HR = %.2f (%.2f-%.2f)",
  hazard_ratio,
  hazard_ratio_lower,
  hazard_ratio_upper
)

statistics <- data.frame(
  contrast = "Higher risk vs lower risk",
  hazard_ratio = hazard_ratio,
  hazard_ratio_lower_95 = hazard_ratio_lower,
  hazard_ratio_upper_95 = hazard_ratio_upper,
  cox_wald_p = cox_wald_p,
  cox_proportional_hazards_p = cox_ph_p,
  stratified_logrank_chi_square = logrank_statistics$chi_square,
  stratified_logrank_p = logrank_p
)


## 5. Calculate the number-at-risk table
risk_times <- c(0, 2, 4, 6, 8, 10)

risk_table <- expand.grid(
  time_years = risk_times,
  risk_group = levels(risk_mean$risk_group),
  stringsAsFactors = FALSE
)

risk_table$n_at_risk <- mapply(
  function(time_value, group_name) {
    sum(
      risk_mean$risk_group == group_name &
      risk_mean$time >= time_value * 365.25
    )
  },
  risk_table$time_years,
  risk_table$risk_group
)

risk_table$risk_group <- factor(
  risk_table$risk_group,
  levels = c("Higher risk", "Lower risk")
)


## 6. Export source data
write.table(
  risk_mean,
  "Source_Data/Figure3/Fig3c_patient_groups.tsv",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

write.table(
  km_curve,
  "Source_Data/Figure3/Fig3c_km_source_data.tsv",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

write.table(
  censor_marks,
  "Source_Data/Figure3/Fig3c_censor_marks.tsv",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

write.table(
  risk_table,
  "Source_Data/Figure3/Fig3c_risk_table_source_data.tsv",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

write.table(
  statistics,
  "Source_Data/Figure3/Fig3c_statistics.tsv",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)


## 7. Make Fig. 3c

write.table(cox_ph_test$table, "Source_Data/Figure3/Fig3c_proportional_hazards.tsv", sep="\t", quote=FALSE, col.names=NA)
