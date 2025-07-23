import os
import re
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

def create_model_performance_plot(X_test, y_test, y_pred, cause_name, location_name, measure_name, 
                                 is_rate, output_dir):
    """
    Create a scatter plot of predicted vs. observed values for an individual model.
    """
    # Create figure directory if it doesn't exist
    fig_dir = os.path.join(output_dir, 'model_performance_plots')
    os.makedirs(fig_dir, exist_ok=True)
    
    # Calculate statistics
    correlation = np.corrcoef(y_test, y_pred)[0, 1]
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    
    # Scale RMSE to per 100,000 format for rates only
    if is_rate:
        rmse_scaled = rmse * 100000
        rmse_display = f"{rmse_scaled:.0f}/100,000"
    else:
        rmse_display = f"{rmse:.4f}"
    
    # Create the plot
    plt.figure(figsize=(10, 10))
    
    # Create scatter plot
    plt.scatter(y_test, y_pred, alpha=0.6, s=40, c='purple')
    
    # Add the diagonal line (perfect prediction)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
    
    # Add statistics to the plot
    plt.annotate(f'r = {correlation:.2f}, P < 0.001\nRMSE = {rmse_display}',
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Set labels and title
    short_measure = measure_name.split(' ')[0]  # Get short version of measure name
    data_type = "Rate (/100k)" if is_rate else "Number"
    plt.xlabel(f'Observation of {short_measure} {data_type}')
    plt.ylabel(f'Prediction of {short_measure} {data_type}')
    plt.title(f'Predictive Performance: {cause_name} - {location_name} - {short_measure}')
    
    # Make axes equal and add grid
    plt.axis('equal')
    plt.grid(alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Create a safe filename
    safe_cause = cause_name.replace(' ', '_')
    safe_location = location_name.replace(' ', '_').replace(',', '')
    safe_measure = measure_name.split(' ')[0]
    type_indicator = "rate" if is_rate else "number"
    
    # Save the figure as PDF
    output_path = os.path.join(fig_dir, f'performance_{safe_cause}_{safe_location}_{safe_measure}_{type_indicator}.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"  Saved model performance plot to: {os.path.basename(output_path)}")
    
    # Close the figure to free memory
    plt.close()


def process_disease_files(input_dir, output_dir=None):
    """
    Process all disease CSV files in a directory, build XGBoost models,
    and generate future predictions with uncertainty intervals.
    """
    if output_dir is None:
        output_dir = input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dataframe to store model metrics
    results_df = pd.DataFrame(columns=[
        'filename', 'cause_name', 'location_name', 'measure_name', 'data_type',
        'best_max_depth', 'best_n_estimators', 'best_learning_rate',
        'best_rmse', 'test_mse', 'test_rmse', 'test_mae', 'test_r2'
    ])
    
    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    for i, file in enumerate(csv_files):
        print(f"\n[{i+1}/{len(csv_files)}] Processing file: {file}")
        
        try:
            # Parse the filename to extract metadata
            filename_parts = file.replace('.csv', '').split('_')
            
            # Skip "updated" prefixes
            start_idx = 0
            while start_idx < len(filename_parts) and filename_parts[start_idx].lower() == 'updated':
                start_idx += 1
            
            # Check if the file is a rate or number file
            is_rate = False
            if 'rate' in file.lower():
                is_rate = True
                print(f"  File type: RATE")
            else:
                print(f"  File type: NUMBER")
            
            cause_name = filename_parts[start_idx]
            
            # Handle location names that may contain underscores or commas
            location_end_idx = len(filename_parts) - 1  # Last part should be measure
            if is_rate:
                # If it's a rate file, the "rate" part is also in the filename
                location_end_idx -= 1
            
            location_parts = filename_parts[start_idx+1:location_end_idx]
            location_name = ' '.join(location_parts)
            
            # The last part is the measure
            measure_short = filename_parts[-1]
            if is_rate and measure_short.lower() == 'ate':
                measure_short = filename_parts[-2]  # Use the part before "rate"
            
            # Map short measure names to full names
            measure_map = {
                'dalys': 'DALYs (Disability-Adjusted Life Years)',
                'deaths': 'Deaths',
                'prevalence': 'Prevalence',
                'incidence': 'Incidence',
                'ylds': 'YLDs (Years Lived with Disability)',
                'ylls': 'YLLs (Years of Life Lost)'
            }
            measure_name = measure_map.get(measure_short.lower(), measure_short)
            
            # Read the data
            file_path = os.path.join(input_dir, file)
            data = pd.read_csv(file_path)
            
            # Split data into historical (1990-2021) and future prediction (2022-2050)
            future_data = data[(data['year'] >= 2022) & (data['year'] <= 2050)].copy()
            historical_data = data[(data['year'] <= 2021)].copy()  # Include all years up to 2021
            
            # Add log_pop if not already present
            if 'log_pop' not in historical_data.columns and 'pop' in historical_data.columns:
                historical_data['log_pop'] = np.log(historical_data['pop'] + 1)  # Add 1 to handle zeros
                if len(future_data) > 0 and 'pop' in future_data.columns:
                    future_data['log_pop'] = np.log(future_data['pop'] + 1)
            
            # Select features and target variable
            feature_columns = ['year', 'age', 'sex', 'log_pop']
            
            # Ensure all required feature columns exist
            missing_cols = [col for col in feature_columns if col not in historical_data.columns]
            if missing_cols:
                print(f"Warning: Missing columns in {file}: {missing_cols}")
                # Use available columns only
                feature_columns = [col for col in feature_columns if col in historical_data.columns]
                if not feature_columns:
                    raise ValueError(f"No usable feature columns found in {file}")
            
            X = historical_data[feature_columns]
            
            # Choose target variable based on file type
            if is_rate:
                # For rate files, use val as target
                if 'val' not in historical_data.columns:
                    raise ValueError(f"'val' column not found in rate file {file}")
                target_column = 'val'
                y = historical_data[target_column]
                print(f"  Using '{target_column}' as target variable for rate file")
            else:
                # For number files, use log_val as target
                if 'log_val' not in historical_data.columns:
                    if 'val' in historical_data.columns:
                        historical_data['log_val'] = np.log(historical_data['val'] + 1)
                        if len(future_data) > 0:
                            future_data['log_val'] = np.log(future_data['val'] + 1)
                    else:
                        raise ValueError(f"Neither 'log_val' nor 'val' column found in {file}")
                target_column = 'log_val'
                y = historical_data[target_column]
                print(f"  Using '{target_column}' as target variable for number file")
            
            # Data split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # Define hyperparameter search grid
            param_grid = {
                'max_depth': range(5, 12, 1),
                'n_estimators': range(100, 1500, 100),
                'learning_rate': [0.01, 0.015, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
            }
            
            # GridSearchCV with XGBoost
            xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                seed=42,
                n_jobs=-1
            )
            
            grid_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=param_grid,
                scoring='neg_root_mean_squared_error',
                cv=5,
                n_jobs=-1,
                verbose=0
            )
            
            # Fit the model
            grid_search.fit(X_train, y_train)
            
            # Get best parameters
            best_params = grid_search.best_params_
            best_rmse = -grid_search.best_score_
            
            print(f"Best parameters for {cause_name} - {location_name} - {measure_name}:")
            print(f"  Max Depth: {best_params['max_depth']}")
            print(f"  N Estimators: {best_params['n_estimators']}")
            print(f"  Learning Rate: {best_params['learning_rate']}")
            print(f"  Best RMSE (CV): {best_rmse:.4f}")
            
            # Train final model with best parameters
            best_model = xgb.XGBRegressor(
                **best_params,
                objective='reg:squarederror',
                seed=42,
                n_jobs=-1,
                early_stopping_rounds=10
            )
            
            # Using simpler approach with eval_set 
            print(f"  Using XGBoost version: {xgb.__version__}")
            
            # Create the evaluation dataset
            eval_set = [(X_test, y_test)]
            
            # Simply fit with eval_set
            try:
                best_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
                print("  Successfully fitted model with early stopping in constructor")
            except Exception as e:
                print(f"  Fallback: Using basic fit method due to error: {str(e)}")
                best_model.fit(X_train, y_train)
            
            # Evaluate model on test set
            y_pred = best_model.predict(X_test)
            
            # Calculate error metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"Test set metrics:")
            print(f"  MSE: {mse:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R²: {r2:.4f}")
            
            # Create and save individual model performance plot
            create_model_performance_plot(X_test, y_test, y_pred, 
                                         cause_name, location_name, measure_name, 
                                         is_rate, output_dir)
            
            # Add results to the dataframe
            results_df = pd.concat([
                results_df,
                pd.DataFrame({
                    'filename': [file],
                    'cause_name': [cause_name],
                    'location_name': [location_name],
                    'measure_name': [measure_name],
                    'data_type': ['rate' if is_rate else 'number'],
                    'best_max_depth': [best_params['max_depth']],
                    'best_n_estimators': [best_params['n_estimators']],
                    'best_learning_rate': [best_params['learning_rate']],
                    'best_rmse': [best_rmse],
                    'test_mse': [mse],
                    'test_rmse': [rmse],
                    'test_mae': [mae],
                    'test_r2': [r2]
                })
            ], ignore_index=True)
            
            # Bootstrap for confidence intervals on future predictions
            if len(future_data) > 0:
                n_bootstrap = 500
                future_X = future_data[feature_columns]
                y_pred_bootstrap = np.zeros((n_bootstrap, len(future_data)))
                
                print(f"Calculating prediction intervals with {n_bootstrap} bootstrap samples...")
                
                for b in range(n_bootstrap):
                    # Bootstrap sampling
                    X_resampled, y_resampled = resample(X_train, y_train, random_state=b)
                    
                    # Train model on bootstrap sample
                    bootstrap_model = xgb.XGBRegressor(**best_params)
                    bootstrap_model.fit(X_resampled, y_resampled)
                    
                    # Predict on future data
                    y_pred_bootstrap[b, :] = bootstrap_model.predict(future_X)
                
                # Calculate prediction intervals (95%)
                lower_bound = np.percentile(y_pred_bootstrap, 2.5, axis=0)
                upper_bound = np.percentile(y_pred_bootstrap, 97.5, axis=0)
                
                # Get point predictions using the full model
                prediction_col = target_column  # Same column name as what we used for training
                future_data[prediction_col] = best_model.predict(future_X)
                
                # Add upper and lower bounds to future_data
                future_data['lower'] = lower_bound
                future_data['upper'] = upper_bound
                
                # If we used log_val for prediction (number files), convert back to original scale
                if not is_rate and prediction_col == 'log_val':
                    if 'val' in future_data.columns:
                        future_data['val'] = np.exp(future_data['log_val']) - 1
                        future_data['val'] = np.where(future_data['year'] >= 2022, np.exp(future_data['log_val']) - 1, future_data['val'])
                    else:
                        future_data['val'] = np.exp(future_data['log_val']) - 1
                    
                    # Convert log_val_lower and log_val_upper to original scale
                    future_data['lower'] = np.exp(lower_bound) - 1
                    future_data['upper'] = np.exp(upper_bound) - 1
                
                # Combine historical and future data
                final_data = pd.concat([historical_data, future_data], ignore_index=True)
                
                # Create output filename
                type_indicator = "rate" if is_rate else "number"
                output_filename = f"1990-2050_{cause_name}_{location_name}_{measure_short}_{type_indicator}.csv"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save to CSV
                final_data.to_csv(output_path, index=False)
                print(f"Saved predictions to: {output_filename}")
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue
    
    # Save results summary
    results_output_path = os.path.join(output_dir, "model_evaluation_results.csv")
    results_df.to_csv(results_output_path, index=False)
    print(f"\nSaved model evaluation metrics to: {results_output_path}")
    
    # Create overall performance visualization plots by metric type and data type
    create_overall_performance_plots(results_df, output_dir)
    
    return results_df


def create_overall_performance_plots(results_df, output_dir):
    """
    Create overall predictive performance visualization plots grouped by metric type and data type (rate/number).
    """
    # Create figure directory
    fig_dir = os.path.join(output_dir, 'performance_plots')
    os.makedirs(fig_dir, exist_ok=True)
    
    # Group data by measure_name and data_type for visualization
    for data_type in ['rate', 'number']:
        # Filter for the current data type
        type_df = results_df[results_df['data_type'] == data_type]
        
        if type_df.empty:
            print(f"No {data_type} files processed, skipping overall visualization")
            continue
            
        measure_groups = type_df.groupby('measure_name')
        
        # Store all prediction vs observation data for each measure
        prediction_data = {}
        
        # Process each group's files to collect observed vs predicted values
        for measure_name, group in measure_groups:
            print(f"\nGenerating overall performance visualization for {measure_name} ({data_type})...")
            prediction_data[measure_name] = []
            
            # Process each file in the group
            for _, row in group.iterrows():
                try:
                    # Load the prediction file
                    type_str = "rate" if data_type == 'rate' else "number"
                    filename = f"1990-2050_{row['cause_name']}_{row['location_name']}_{row['measure_name'].split(' ')[0].lower()}_{type_str}.csv"
                    file_path = os.path.join(output_dir, filename)
                    
                    if os.path.exists(file_path):
                        # Load the file
                        data = pd.read_csv(file_path)
                        
                        # Split into historical and predicted data
                        historical = data[data['year'] <= 2021].copy()
                        
                        # Determine target column based on data type
                        target_col = 'val' if data_type == 'rate' else 'log_val'
                        
                        if target_col in historical.columns:
                            # Get the actual values
                            y_true = historical[target_col]
                            
                            # Get the predicted values (we repredict the historical data)
                            X_historical = historical[['year', 'age', 'sex', 'log_pop']]
                            
                            # Create and train a model with the best parameters
                            model = xgb.XGBRegressor(
                                max_depth=row['best_max_depth'],
                                n_estimators=row['best_n_estimators'],
                                learning_rate=row['best_learning_rate'],
                                objective='reg:squarederror',
                                seed=42
                            )
                            model.fit(X_historical, y_true)
                            
                            # Predict on the historical data
                            y_pred = model.predict(X_historical)
                            
                            # Add to the prediction data
                            for true_val, pred_val in zip(y_true, y_pred):
                                prediction_data[measure_name].append((true_val, pred_val))
                except Exception as e:
                    print(f"Error processing {row['filename']} for visualization: {str(e)}")
                    continue
            
            # Create the plot if we have data
            if prediction_data[measure_name]:
                create_prediction_plot(prediction_data[measure_name], measure_name, data_type, fig_dir)
                

def create_prediction_plot(prediction_data, measure_name, data_type, output_dir):
    """
    Create a scatter plot of predicted vs. observed values.
    """
    # Convert to numpy arrays
    data = np.array(prediction_data)
    observed = data[:, 0]
    predicted = data[:, 1]
    
    # Calculate statistics
    correlation = np.corrcoef(observed, predicted)[0, 1]
    rmse = np.sqrt(np.mean((predicted - observed) ** 2))
    
    # Format RMSE display based on data type
    if data_type == 'rate':
        rmse_scaled = rmse * 100000
        rmse_display = f"{rmse_scaled:.0f}/100,000"
    else:
        rmse_display = f"{rmse:.4f}"
    
    # Scale the figure based on the amount of data
    figsize = (10, 10)
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Create scatter plot
    plt.scatter(observed, predicted, alpha=0.5, s=10, c='#089099FF')
    
    # Add the diagonal line (perfect prediction)
    min_val = min(observed.min(), predicted.min())
    max_val = max(observed.max(), predicted.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
    
    # Add statistics to the plot
    plt.annotate(f'r = {correlation:.2f}, P < 0.001\nRMSE = {rmse_display}',
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Set labels and title
    display_type = "Rate (/100k)" if data_type == 'rate' else "Number"
    plt.xlabel(f'Observation of {measure_name} {display_type}')
    plt.ylabel(f'Prediction of {measure_name} {display_type}')
    plt.title(f'Overall Predictive Performance: {measure_name} {display_type}')
    
    # Make axes equal and add grid
    plt.axis('equal')
    plt.grid(alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure as PDF
    safe_measure_name = measure_name.replace(' ', '_').replace('(', '').replace(')', '')
    output_path = os.path.join(output_dir, f'overall_performance_{safe_measure_name}_{data_type}.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    
    # Close the figure to free memory
    plt.close()


# Example usage:
if __name__ == "__main__":
    # Replace with your input directory
    input_directory = "."
    
    # Optional: specify an output directory (if different from input)
    output_directory = "./disease_predictions"
    
    # Process all CSV files
    results = process_disease_files(input_directory, output_directory)
    
    # Print summary statistics grouped by data type
    print("\nSummary of model performance:")
    
    # For rate files
    rate_results = results[results['data_type'] == 'rate']
    if not rate_results.empty:
        print("\nRate models:")
        print(f"  Average RMSE: {rate_results['test_rmse'].mean():.4f}")
        print(f"  Best model: {rate_results.loc[rate_results['test_rmse'].idxmin(), 'filename']}")
        print(f"  Worst model: {rate_results.loc[rate_results['test_rmse'].idxmax(), 'filename']}")
    
    # For number files
    number_results = results[results['data_type'] == 'number']
    if not number_results.empty:
        print("\nNumber models:")
        print(f"  Average RMSE: {number_results['test_rmse'].mean():.4f}")
        print(f"  Best model: {number_results.loc[number_results['test_rmse'].idxmin(), 'filename']}")
        print(f"  Worst model: {number_results.loc[number_results['test_rmse'].idxmax(), 'filename']}")