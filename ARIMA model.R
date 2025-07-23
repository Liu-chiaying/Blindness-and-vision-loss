

a <- read.csv(file ='super-region+SDI.csv')
head(a)
unique(a$age_name)
unique(a$location_name)
unique(filtered_data$metric_name)

target_ages <- c(   
  "Age-standardized")
                 # "<5 years", "5-9 years", "10-14 years", "15-19 years", "20-24 years", 
                 #                     "25-29 years", "30-34 years", "35-39 years", "40-44 years", "45-49 years", "50-54 years", "55-59 years",
                 #                     "60-64 years", "65-69 years", "70-74 years", "75-79 years", "80-84 years", "85-89 years", "90-94 years", "95+ years")


filtered_data <- a[a$age_name %in% target_ages, ]


split_data_by_categories <- function(data, output_dir = "split_files", save_files = TRUE, 
                                     deduplicate = TRUE, dedup_method = "first") {

  if (save_files && !dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
    message(paste("创建输出目录:", output_dir))
  }
  

  if (deduplicate) {
    original_rows <- nrow(data)
    message("开始去除重复数据...")
    data <- remove_location_id_duplicates(data, method = dedup_method)
    message(paste("去重前行数:", original_rows, ", 去重后行数:", nrow(data)))
  }
  

  measures <- unique(data$measure_name)
  locations <- unique(data$location_name)
  metrics <- unique(data$metric_name)

  total_combinations <- length(measures) * length(locations) * length(metrics)
  message(paste("将处理", total_combinations, "个数据组合"))
  

  file_log <- data.frame(
    measure = character(),
    location = character(),
    metric = character(),
    file_name = character(),
    row_count = integer(),
    stringsAsFactors = FALSE
  )
  

  data_subsets <- list()
  

  counter <- 0
  

  for (measure in measures) {
    for (location in locations) {
      for (metric in metrics) {

        counter <- counter + 1
        

        subset_data <- subset(data, 
                              measure_name == measure & 
                                location_name == location & 
                                metric_name == metric)
        

        if (nrow(subset_data) == 0) {
          message(paste("跳过空子集:", measure, "-", location, "-", metric))
          next
        }
        

        subset_id <- paste(measure, location, metric, sep = "_||_")
        

        data_subsets[[subset_id]] <- subset_data
        
        if (save_files) {
  
          safe_measure <- gsub("[^a-zA-Z0-9]", "_", measure)
          safe_location <- gsub("[^a-zA-Z0-9]", "_", location)
          safe_metric <- gsub("[^a-zA-Z0-9]", "_", metric)
          

          file_name <- paste0(
            safe_measure, "_", 
            safe_location, "_", 
            safe_metric, 
            ".csv"
          )
          

          file_path <- file.path(output_dir, file_name)
          

          write.csv(subset_data, file = file_path, row.names = FALSE)
        } else {

          file_name <- ""
        }
        

        file_log <- rbind(file_log, data.frame(
          measure = measure,
          location = location,
          metric = metric,
          file_name = file_name,
          row_count = nrow(subset_data),
          stringsAsFactors = FALSE
        ))
        

        if (counter %% 5 == 0 || counter == total_combinations) {
          message(paste("已处理", counter, "个组合，共", total_combinations, "个 (",
                        round(counter/total_combinations*100), "%)"))
        }
      }
    }
  }
  

  if (save_files) {
    log_file_path <- file.path(output_dir, "split_files_log.csv")
    write.csv(file_log, file = log_file_path, row.names = FALSE)
    message(paste("文件拆分完成! 日志已保存至:", log_file_path))
  }
  

  result <- list(
    data_subsets = data_subsets,  # 包含所有子集数据的列表
    file_log = file_log,          # 文件日志数据框
    measures = measures,          # 所有唯一的 measure_name 值
    locations = locations,        # 所有唯一的 location_name 值
    metrics = metrics             # 所有唯一的 metric_name 值
  )
  
  return(result)
}


generate_and_merge_future_predictions <- function(split_result, output_dir = "merged_data", 
                                                  years = 2022:2050, save_files = TRUE) {

  data_subsets <- split_result$data_subsets
  

  merged_data_subsets <- list()
  

  merged_file_log <- data.frame(
    subset_id = character(),
    file_name = character(),
    historical_rows = integer(),
    future_rows = integer(),
    total_rows = integer(),
    stringsAsFactors = FALSE
  )
  

  if (save_files && !dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
    message(paste("创建输出目录:", output_dir))
  }
  

  total_subsets <- length(data_subsets)
  message(paste("将为", total_subsets, "个数据子集生成并合并未来预测框架"))
  

  counter <- 0
  

  for (subset_id in names(data_subsets)) {

    counter <- counter + 1
    

    historical_data <- data_subsets[[subset_id]]
    

    value_cols <- c("year", "val", "upper", "lower")
    template_cols <- setdiff(names(historical_data), value_cols)
    

    if (requireNamespace("dplyr", quietly = TRUE)) {
      template_rows <- dplyr::distinct(historical_data[, template_cols])
    } else {
      template_rows <- historical_data[!duplicated(historical_data[, template_cols]), template_cols]
    }
    

    future_data <- data.frame()
    
    for (i in 1:nrow(template_rows)) {
      row_template <- template_rows[i, ]
      

      for (year in years) {
        new_row <- row_template
        new_row$year <- year
        new_row$val <- NA
        new_row$upper <- NA
        new_row$lower <- NA
        
        future_data <- rbind(future_data, new_row)
      }
    }
    

    merged_data <- rbind(historical_data, future_data)
    

    merged_data <- merged_data[order(merged_data$year), ]

    merged_data_subsets[[subset_id]] <- merged_data
    
    if (save_files) {
      id_parts <- strsplit(subset_id, "_\\|\\|_")[[1]]
      measure <- id_parts[1]
      location <- id_parts[2]
      metric <- id_parts[3]
      

      safe_measure <- gsub("[^a-zA-Z0-9]", "_", measure)
      safe_location <- gsub("[^a-zA-Z0-9]", "_", location)
      safe_metric <- gsub("[^a-zA-Z0-9]", "_", metric)
      

      file_name <- paste0(
        safe_measure, "_", 
        safe_location, "_", 
        safe_metric, 
        "_merged.csv"
      )

      file_path <- file.path(output_dir, file_name)
      
      write.csv(merged_data, file = file_path, row.names = FALSE)
      

      merged_file_log <- rbind(merged_file_log, data.frame(
        subset_id = subset_id,
        file_name = file_name,
        historical_rows = nrow(historical_data),
        future_rows = nrow(future_data),
        total_rows = nrow(merged_data),
        stringsAsFactors = FALSE
      ))
    }
    
    if (counter %% 5 == 0 || counter == total_subsets) {
      message(paste("已处理", counter, "个子集，共", total_subsets, "个 (",
                    round(counter/total_subsets*100), "%)"))
    }
  }
  

  if (save_files) {
    log_file_path <- file.path(output_dir, "merged_files_log.csv")
    write.csv(merged_file_log, file = log_file_path, row.names = FALSE)
    message(paste("历史数据与未来预测框架合并完成! 日志已保存至:", log_file_path))
  }
  

  result <- list(
    merged_data_subsets = merged_data_subsets,  
    merged_file_log = merged_file_log,          
    historical_subsets = data_subsets           
  )
  
  return(result)
}


merged_result <- generate_and_merge_future_predictions(
  split_result = result,
  output_dir = "merged_prediction_data",
  years = 2022:2050
)



process_with_bootstrap <- function(merged_data_list, n_bootstrap = 500, output_dir = "bootstrap_forecast_results") {

  required_packages <- c("forecast", "dplyr", "tidyr", "ggplot2")
  for (pkg in required_packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      install.packages(pkg)
    }
    library(pkg, character.only = TRUE)
  }
  

  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
    message(paste("创建输出目录:", output_dir))
  }
  

  processed_results <- list()
  

  results_log <- data.frame(
    subset_id = character(),
    file_name = character(),
    metrics_file = character(),
    stringsAsFactors = FALSE
  )
  

  merged_data_subsets <- merged_data_list$merged_data_subsets
  total_subsets <- length(merged_data_subsets)
  message(paste("将处理", total_subsets, "个数据子集"))
  

  counter <- 0
  

  for (subset_id in names(merged_data_subsets)) {

    counter <- counter + 1
    
    current_data <- merged_data_subsets[[subset_id]]
    

    current_data$year <- as.numeric(as.character(current_data$year))
    

    current_data <- current_data %>% arrange(year)
    

    historical_data <- current_data[current_data$year <= 2021, ]
    future_data <- current_data[current_data$year > 2021, ]
    
    message(paste("处理子集", counter, "/", total_subsets, ":", subset_id))
    message(paste("  历史数据行数:", nrow(historical_data)))
    message(paste("  未来数据框架行数:", nrow(future_data)))
    

    group_vars <- c("age_name", "sex_name")
    

    if (!all(group_vars %in% colnames(historical_data))) {
      missing_vars <- group_vars[!group_vars %in% colnames(historical_data)]
      message(paste("  警告: 数据中缺少分组变量:", paste(missing_vars, collapse=", ")))
      

      group_vars <- group_vars[group_vars %in% colnames(historical_data)]
      
      if (length(group_vars) == 0) {
        message("  错误: 没有可用的分组变量，跳过此子集")
        next
      }
    }
    

    group_expr <- as.list(historical_data[, group_vars, drop=FALSE])
    

    group_list <- split(historical_data, group_expr)
    

    group_results <- list()
    for (group_name in names(group_list)) {
      data_group <- group_list[[group_name]]
      

      data_group <- data_group %>% arrange(year)
      

      group_info <- list()
      for (var in group_vars) {
        group_info[[var]] <- unique(data_group[[var]])
      }
      

      val_data <- data_group$val
      if (any(is.na(val_data))) {
        message(paste("  警告: 分组", group_name, "中val列包含NA值，将被移除"))
        val_data[is.na(val_data)] <- mean(val_data, na.rm = TRUE)  # 用均值替换NA
      }
      

      ts_data <- ts(val_data, start = min(data_group$year), frequency = 1)
      

      fit <- tryCatch({
        auto.arima(ts_data)
      }, error = function(e) {
        message(paste("  警告: 分组", group_name, "ARIMA模型拟合失败:", e$message))

        tryCatch({
          forecast::Arima(ts_data, order = c(0,1,0))
        }, error = function(e2) {
          message(paste("  严重错误: 备选ARIMA模型也失败:", e2$message))

          model <- list(
            fitted = ts_data,
            residuals = ts_data - mean(ts_data),
            method = "Mean (fallback)"
          )
          class(model) <- "forecast"
          return(model)
        })
      })
      

      fitted_values <- fitted(fit)
      

      future_years <- 2022:2050
      h_value <- length(future_years)
      
      forecast_result <- tryCatch({
        forecast(fit, h = h_value)
      }, error = function(e) {
        message(paste("  警告: 分组", group_name, "预测失败:", e$message))

        last_value <- tail(ts_data, 1)
        mean_values <- rep(last_value, h_value)
        forecast_obj <- list(
          mean = mean_values,
          lower = matrix(mean_values * 0.9, ncol = 2),
          upper = matrix(mean_values * 1.1, ncol = 2),
          method = "Constant (fallback)"
        )
        class(forecast_obj) <- "forecast"
        return(forecast_obj)
      })
      

      residuals_vector <- as.vector(residuals(fit))
      

      residuals_vector <- residuals_vector[!is.na(residuals_vector)]
      

      if (length(residuals_vector) < 2) {
        message(paste("  警告: 分组", group_name, "中残差不足，使用简单方法估计置信区间"))
        residuals_vector <- c(-0.1, 0, 0.1) * mean(abs(val_data))  # 创建一些人工残差
      }
      

      forecast_mean <- as.vector(forecast_result$mean)
      
      bootstrap_forecasts <- matrix(0, nrow = n_bootstrap, ncol = length(forecast_mean))
      

      for (i in 1:n_bootstrap) {

        bootstrap_residuals <- sample(residuals_vector, length(forecast_mean), replace = TRUE)
        

        bootstrap_forecasts[i, ] <- forecast_mean + bootstrap_residuals
      }
      

      lower_bound <- apply(bootstrap_forecasts, 2, function(x) quantile(x, 0.025))
      upper_bound <- apply(bootstrap_forecasts, 2, function(x) quantile(x, 0.975))
      

      group_results[[group_name]] <- list(
        group_info = group_info,
        fit = fit,
        forecast = forecast_result,
        bootstrap_lower = lower_bound,
        bootstrap_upper = upper_bound
      )
    }
    

    combined_results <- data.frame()
    
    for (group_name in names(group_results)) {
      current_result <- group_results[[group_name]]
      

      match_cond <- list()
      for (var in names(current_result$group_info)) {
        match_cond[[var]] <- current_result$group_info[[var]]
      }
      

      match_df <- as.data.frame(match_cond)
      

      group_historical <- merge(historical_data, match_df)
      

      group_future <- merge(future_data, match_df)
      

      group_historical <- group_historical %>% arrange(year)
      group_future <- group_future %>% arrange(year)
      
      if (nrow(group_future) == 0) {
        message(paste("  警告: 分组", group_name, "在未来数据框架中没有记录"))
        next
      }
      

      forecast_values <- as.numeric(current_result$forecast$mean)
      

      if (length(forecast_values) != nrow(group_future)) {
        message(paste("  警告: 分组", group_name, "预测值数量与未来年份不匹配"))
        message(paste("  预测值长度:", length(forecast_values), "未来行数:", nrow(group_future)))
        

        if (length(forecast_values) > nrow(group_future)) {

          forecast_values <- forecast_values[1:nrow(group_future)]
          if (length(current_result$bootstrap_lower) >= nrow(group_future)) {
            current_result$bootstrap_lower <- current_result$bootstrap_lower[1:nrow(group_future)]
            current_result$bootstrap_upper <- current_result$bootstrap_upper[1:nrow(group_future)]
          } else {

            current_result$bootstrap_lower <- c(
              current_result$bootstrap_lower, 
              rep(tail(current_result$bootstrap_lower, 1), nrow(group_future) - length(current_result$bootstrap_lower))
            )
            current_result$bootstrap_upper <- c(
              current_result$bootstrap_upper, 
              rep(tail(current_result$bootstrap_upper, 1), nrow(group_future) - length(current_result$bootstrap_upper))
            )
          }
        } else {

          last_value <- forecast_values[length(forecast_values)]
          forecast_values <- c(forecast_values, rep(last_value, nrow(group_future) - length(forecast_values)))
          

          if (length(current_result$bootstrap_lower) < nrow(group_future)) {
            last_lower <- current_result$bootstrap_lower[length(current_result$bootstrap_lower)]
            current_result$bootstrap_lower <- c(
              current_result$bootstrap_lower, 
              rep(last_lower, nrow(group_future) - length(current_result$bootstrap_lower))
            )
            
            last_upper <- current_result$bootstrap_upper[length(current_result$bootstrap_upper)]
            current_result$bootstrap_upper <- c(
              current_result$bootstrap_upper, 
              rep(last_upper, nrow(group_future) - length(current_result$bootstrap_upper))
            )
          }
        }
      }
      

      arima_params <- tryCatch({
        p <- 0
        d <- 0
        q <- 0
        aic <- NA
        

        if (inherits(current_result$fit, "Arima")) {
          arima_order <- arimaorder(current_result$fit)
          p <- arima_order[1]
          d <- arima_order[2]
          q <- arima_order[3]
          aic <- current_result$fit$aic
        }
        
        list(p = p, d = d, q = q, aic = aic)
      }, error = function(e) {
        message(paste("  警告: 无法获取ARIMA参数:", e$message))
        list(p = 0, d = 0, q = 0, aic = NA)
      })
      

      historical_fitted <- tryCatch({
        as.numeric(fitted(current_result$fit))
      }, error = function(e) {
        message(paste("  警告: 无法获取拟合值:", e$message))
        rep(mean(group_historical$val), nrow(group_historical))
      })
      

      if (length(historical_fitted) != nrow(group_historical)) {
        message(paste("  警告: 拟合值长度不匹配，拟合值:", length(historical_fitted), "历史数据:", nrow(group_historical)))
        
        if (length(historical_fitted) > nrow(group_historical)) {
          historical_fitted <- historical_fitted[1:nrow(group_historical)]
        } else {
          historical_fitted <- c(historical_fitted, rep(mean(historical_fitted), nrow(group_historical) - length(historical_fitted)))
        }
      }
      

      hist_df <- group_historical
      hist_df$fitted_value <- historical_fitted
      hist_df$type <- "actual"
      hist_df$p <- arima_params$p
      hist_df$d <- arima_params$d
      hist_df$q <- arima_params$q
      hist_df$AIC <- arima_params$aic
      
 
      forecast_df <- group_future
      forecast_df$val <- forecast_values  
      forecast_df$fitted_value <- forecast_values
      

      if (length(current_result$bootstrap_lower) != nrow(forecast_df) || 
          length(current_result$bootstrap_upper) != nrow(forecast_df)) {
        message(paste("  警告: Bootstrap区间长度不匹配，调整中..."))
        

        if (length(current_result$bootstrap_lower) > nrow(forecast_df)) {
          current_result$bootstrap_lower <- current_result$bootstrap_lower[1:nrow(forecast_df)]
        } else {
          last_lower <- ifelse(length(current_result$bootstrap_lower) > 0, 
                               tail(current_result$bootstrap_lower, 1), 
                               0.9 * mean(forecast_values))
          current_result$bootstrap_lower <- c(current_result$bootstrap_lower, 
                                              rep(last_lower, nrow(forecast_df) - length(current_result$bootstrap_lower)))
        }
        

        if (length(current_result$bootstrap_upper) > nrow(forecast_df)) {
          current_result$bootstrap_upper <- current_result$bootstrap_upper[1:nrow(forecast_df)]
        } else {
          last_upper <- ifelse(length(current_result$bootstrap_upper) > 0, 
                               tail(current_result$bootstrap_upper, 1), 
                               1.1 * mean(forecast_values))
          current_result$bootstrap_upper <- c(current_result$bootstrap_upper, 
                                              rep(last_upper, nrow(forecast_df) - length(current_result$bootstrap_upper)))
        }
      }
      
      forecast_df$upper <- current_result$bootstrap_upper
      forecast_df$lower <- current_result$bootstrap_lower
      forecast_df$type <- "forecast"
      forecast_df$p <- arima_params$p
      forecast_df$d <- arima_params$d
      forecast_df$q <- arima_params$q
      forecast_df$AIC <- arima_params$aic
      

      temp_df <- rbind(hist_df, forecast_df)
      

      temp_df$residual <- ifelse(temp_df$type == "actual", 
                                 temp_df$val - temp_df$fitted_value, 
                                 NA)
      

      combined_results <- rbind(combined_results, temp_df)
    }
    

    if (nrow(combined_results) == 0) {
      message(paste("  警告: 子集", subset_id, "没有产生任何结果，跳过"))
      next
    }
    

    sort_vars <- c(group_vars, "year")
    sort_vars <- sort_vars[sort_vars %in% colnames(combined_results)]
    

    final_result <- combined_results %>%
      arrange(across(all_of(sort_vars)))
    

    historical_result <- final_result[final_result$type == "actual", ]
    
    if (nrow(historical_result) > 0) {
      actual <- historical_result$val
      predicted <- historical_result$fitted_value
      

      valid_indices <- which(!is.na(actual) & !is.na(predicted))
      
      if (length(valid_indices) > 0) {
        metrics_df <- calculate_metrics(actual[valid_indices], predicted[valid_indices])
      } else {
        metrics_df <- data.frame(
          Metric = c("MSE", "RMSE", "MAPE", "SMAPE", "MASE", "R²"),
          Value = rep(NA, 6)
        )
      }
    } else {
      metrics_df <- data.frame(
        Metric = c("MSE", "RMSE", "MAPE", "SMAPE", "MASE", "R²"),
        Value = rep(NA, 6)
      )
    }
    
    processed_results[[subset_id]] <- final_result
    

    if (!is.null(output_dir)) {

      id_parts <- strsplit(subset_id, "_\\|\\|_")[[1]]
      

      if (length(id_parts) >= 1) {
        measure <- id_parts[1]
        location <- if (length(id_parts) >= 2) id_parts[2] else "Unknown"
        metric <- if (length(id_parts) >= 3) id_parts[3] else "Unknown"
      } else {

        measure <- subset_id
        location <- "Unknown"
        metric <- "Unknown"
      }
      
      safe_measure <- gsub("[^a-zA-Z0-9]", "_", measure)
      safe_location <- gsub("[^a-zA-Z0-9]", "_", location)
      safe_metric <- gsub("[^a-zA-Z0-9]", "_", metric)
      
      # 构建文件名
      result_file_name <- paste0(
        safe_measure, "_", 
        safe_location, "_", 
        safe_metric, 
        "_forecast_bootstrap.csv"
      )
      
      metrics_file_name <- paste0(
        safe_measure, "_", 
        safe_location, "_", 
        safe_metric, 
        "_metrics.csv"
      )
      
      # 完整文件路径
      result_file_path <- file.path(output_dir, result_file_name)
      metrics_file_path <- file.path(output_dir, metrics_file_name)
      
      # 保存数据
      write.csv(final_result, file = result_file_path, row.names = FALSE)
      write.csv(metrics_df, file = metrics_file_path, row.names = FALSE)
      
      # 添加到日志
      results_log <- rbind(results_log, data.frame(
        subset_id = subset_id,
        file_name = result_file_name,
        metrics_file = metrics_file_name,
        stringsAsFactors = FALSE
      ))
    }
  
    message(paste("  完成处理子集", counter, "/", total_subsets, " - 结果行数:", nrow(final_result)))
  }
  

  if (!is.null(output_dir)) {
    log_file_path <- file.path(output_dir, "bootstrap_results_log.csv")
    write.csv(results_log, file = log_file_path, row.names = FALSE)
    message(paste("所有处理完成! 日志已保存至:", log_file_path))
  }
  

  final_results <- list(
    processed_data = processed_results,
    results_log = results_log
  )
  
  return(final_results)
}


calculate_metrics <- function(actual, predicted) {
 
  if (!requireNamespace("Metrics", quietly = TRUE)) {
    install.packages("Metrics")
  }
  library(Metrics)
  

  mse <- mse(actual, predicted)
  rmse <- rmse(actual, predicted)
  

  if (any(actual == 0)) {
    valid_indices <- which(actual != 0)
    if (length(valid_indices) > 0) {
      mape <- mape(actual[valid_indices], predicted[valid_indices])
    } else {
      mape <- NA
      warning("所有实际值为0，MAPE无法计算")
    }
  } else {
    mape <- mape(actual, predicted)
  }
  

  smape <- smape(actual, predicted)
  
  
  mae <- mean(abs(actual - predicted))
  if (length(actual) > 1) {
    naive_errors <- abs(actual[2:length(actual)] - actual[1:(length(actual)-1)])
    naive_mae <- mean(naive_errors)
    if (naive_mae > 0) {
      mase <- mae / naive_mae
    } else {
      mase <- NA
      warning("naive_mae为0，MASE无法计算")
    }
  } else {
    mase <- NA
    warning("数据长度不足以计算MASE")
  }
  

  mean_actual <- mean(actual)
  tss <- sum((actual - mean_actual)^2)
  if (tss > 0) {
    rss <- sum((actual - predicted)^2)
    r2 <- 1 - (rss / tss)
  } else {
    r2 <- NA
    warning("所有实际值相同，R²无法计算")
  }
  

  metrics_df <- data.frame(
    Metric = c("MSE", "RMSE", "MAPE", "SMAPE", "MASE", "R²"),
    Value = c(mse, rmse, mape, smape, mase, r2)
  )
  
  return(metrics_df)
}



bootstrap_result <- process_with_bootstrap(
  merged_data_list = deduped_result, 
  n_bootstrap = 500,
  output_dir = "bootstrap_forecast_results"
)


head(names(bootstrap_result$processed_data))


first_result_id <- names(bootstrap_result$processed_data)[1]
first_result <- bootstrap_result$processed_data[[first_result_id]]


head(first_result[first_result$type == "actual", ])


head(first_result[first_result$type == "forecast", ])


library(dplyr)
library(readr)
library(stringr)
library(purrr)

# 定义合并函数
merge_metrics_files <- function(directory_path = ".", output_filename = "all_merged_metrics.csv") {
  
  # 获取所有metrics.csv文件
  metrics_files <- list.files(
    path = directory_path,
    pattern = "metrics\\.csv$",
    full.names = TRUE
  )
  

  cat(sprintf("找到 %d 个metrics文件\n", length(metrics_files)))
  
  if(length(metrics_files) == 0) {
    cat("没有找到metrics.csv文件\n")
    return(NULL)
  }
  

  all_data <- map_dfr(metrics_files, function(file) {

    file_basename <- basename(file)
    cat("处理文件:", file_basename, "\n")
    
    temp_name <- gsub("_metrics\\.csv$", "", file_basename)

    last_underscore_pos <- str_locate_all(temp_name, "_")[[1]]
    if(nrow(last_underscore_pos) > 0) {
      last_pos <- last_underscore_pos[nrow(last_underscore_pos), "start"]

      category <- substr(temp_name, 1, last_pos - 1)
    } else {

      category <- temp_name
    }
    
    cat("提取的类别:", category, "\n")
    
    tryCatch({
      df <- read_csv(file, show_col_types = FALSE)
      
      df <- df %>% 
        mutate(
          Category = category,
          FileName = file_basename
        )
      
      cat("成功读取，行数:", nrow(df), "\n")
      return(df)
    }, error = function(e) {
      cat("处理文件时出错:", e$message, "\n")
      return(NULL)
    })
  })
  
  if(is.null(all_data) || nrow(all_data) == 0) {
    cat("没有数据可合并或合并过程出错\n")
    return(NULL)
  }

  all_data <- all_data %>% select(Category, FileName, everything())
  

  output_file <- file.path(directory_path, output_filename)
  write_csv(all_data, output_file)
  
  cat(sprintf("\n合并完成！\n"))
  cat(sprintf("- 输出文件: %s\n", output_file))
  cat(sprintf("- 总行数: %d\n", nrow(all_data)))
  cat(sprintf("- 总列数: %d\n", ncol(all_data)))
  

  cat("\n包含以下类别:\n")
  categories <- unique(all_data$Category)
  for(cat_name in categories) {
    cat(" - ", cat_name, "\n")
  }
  
  return(all_data)
}


merge_by_file_type <- function(directory_path = ".") {
  

  all_files <- list.files(
    path = directory_path,
    pattern = "metrics\\.csv$",
    full.names = FALSE
  )
  

  file_types <- c()
  for(file in all_files) {
    if(grepl("__", file)) {
      type <- strsplit(file, "__")[[1]][1]
    } else if(grepl("_", file)) {
      type <- strsplit(file, "_")[[1]][1]
    } else {
      type <- gsub("metrics\\.csv$", "", file)
    }
    
    file_types <- c(file_types, type)
  }
  file_types <- unique(file_types)
  
  cat("发现以下文件类型:\n")
  for(type in file_types) {
    cat(" - ", type, "\n")
  }
  
  results <- list()
  for(type in file_types) {
    cat(sprintf("\n处理 %s 类型的文件...\n", type))
    
    # 筛选该类型的文件
    type_files <- all_files[grepl(paste0("^", type), all_files)]
    cat(sprintf("找到 %d 个 %s 类型的文件\n", length(type_files), type))
    
    temp_dir <- file.path(directory_path, paste0("temp_", type))
    dir.create(temp_dir, showWarnings = FALSE)
    
    for(file in type_files) {
      file.copy(
        from = file.path(directory_path, file),
        to = file.path(temp_dir, file),
        overwrite = TRUE
      )
    }
    
    output_file <- paste0("merged_", tolower(type), "_metrics.csv")
    results[[type]] <- merge_metrics_files(temp_dir, output_file)
    
    file.copy(
      from = file.path(temp_dir, output_file),
      to = file.path(directory_path, output_file),
      overwrite = TRUE
    )
    
    unlink(temp_dir, recursive = TRUE)
    
    cat(sprintf("完成 %s 类型文件的合并，输出文件: %s\n", 
                type, file.path(directory_path, output_file)))
  }
  
  return(results)
}
merged_data <- merge_metrics_files()



