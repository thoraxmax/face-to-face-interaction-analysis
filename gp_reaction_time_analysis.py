"""
Gaussian Process Analysis for Reaction Time Data with Censoring

This module implements a Bayesian analysis of reaction time data using 
Gaussian Processes with censoring support (Tobit model).

Author: Adapted for face-to-face interaction analysis repository
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import warnings

# Suppress PyMC warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


def robust_scale(X):
    """Robust median/MAD scaling for outlier resistance."""
    med = np.median(X)
    mad = np.median(np.abs(X - med))
    scale = mad if mad > 1e-8 else np.std(X) if np.std(X) > 1e-8 else 1.0
    return (X - med) / scale, (med, scale)


def fit_participant_sparse_gp(df_p, n_inducing=10, censor_limit=3.1, debug=False):
    """
    Fit a sparse Gaussian Process model to participant reaction time data.
    
    Parameters:
    -----------
    df_p : pd.DataFrame
        Participant data with 'spawnrate', 'RT_censored', and 'censored' columns
    n_inducing : int
        Number of inducing points for sparse GP
    censor_limit : float
        Censoring threshold for reaction times
    debug : bool
        Whether to print debug information
        
    Returns:
    --------
    tuple : (trace, scaler, Xu, error, Xs, y_train)
        trace: MCMC trace if successful, None if failed
        scaler: tuple of (median, scale) for input normalization
        Xu: inducing points for sparse GP
        error: error message if fitting failed, None if successful
        Xs: scaled input values
        y_train: target values
    """
    X = df_p['spawnrate'].values
    y = df_p['RT_censored'].values

    # Robust scaling
    Xs, scaler = robust_scale(X)
    Xs = Xs.astype(np.float32)
    y = y.astype(np.float32)

    # Set up inducing points evenly spaced across the input domain
    Xu = np.linspace(Xs.min(), Xs.max(), n_inducing).reshape(-1, 1).astype(np.float32)

    if debug:
        print("    [fit_participant_gp_with_censoring DEBUG]")
        print("    X median/scale:", scaler)
        print("    Xs shape:", Xs.shape)
        print("    y shape:", y.shape)
        print("    Xs range:", Xs.min(), Xs.max())
        print("    y: ", y)
        print("    Number of data points:", len(X))
        print("    Inducing points shape:", Xu.shape)

    try:
        with pm.Model() as model:
            # GP hyperparameters
            ls = pm.HalfNormal("lengthscale", 1.0)
            cov_func = pm.gp.cov.ExpQuad(1, ls)
            sigma = pm.HalfNormal("sigma", 1.0)
            mean_func = pm.gp.mean.Zero()

            # Use standard Latent GP for simpler implementation
            gp = pm.gp.Latent(cov_func=cov_func, mean_func=mean_func)
            f = gp.prior("f", Xs.reshape(-1, 1))

            # Add censoring constraints using Potential
            uncensored = df_p['censored'].values == 0
            censored = ~uncensored
            
            # For uncensored observations, use normal likelihood
            if np.any(uncensored):
                y_uncensored = y[uncensored]
                f_uncensored = f[uncensored]
                pm.Potential(
                    "uncensored_logp",
                    pm.logp(pm.Normal.dist(mu=f_uncensored, sigma=sigma), y_uncensored)
                )

            # For censored observations, use survival function (1 - CDF)
            if np.any(censored):
                f_censored = f[censored]
                pm.Potential(
                    "censored_logp",
                    pm.logcdf(pm.Normal.dist(mu=f_censored, sigma=sigma), censor_limit)
                )

            # Fit using ADVI
            approx = pm.fit(
                n=1500,
                method="advi",
                progressbar=False,  # Set to False to avoid progress bar issues
                callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-2)],
                obj_optimizer=pm.adam(learning_rate=1e-2)
            )
            trace = approx.sample(500)

        return trace, scaler, Xu, None, Xs, y

    except Exception as e:
        return None, scaler, Xu, str(e), Xs, y


def predict_gp(trace, scaler, Xu, Xs, y_train, X_plot):
    """
    Make predictions using the fitted GP model.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        MCMC trace from fitted model
    scaler : tuple
        (median, scale) for input normalization  
    Xu : np.array
        Inducing points (not used in this simplified version)
    Xs : np.array
        Scaled training inputs
    y_train : np.array
        Training targets
    X_plot : np.array
        Points at which to make predictions
        
    Returns:
    --------
    np.array : Predicted values at X_plot
    """
    X_med, X_scale = scaler
    X_plot_scaled = (X_plot - X_med) / X_scale
    
    # Get mean posterior hyperparameters
    lengthscale_mean = trace.posterior["lengthscale"].mean(dim=["chain", "draw"]).item()
    sigma_mean = trace.posterior["sigma"].mean(dim=["chain", "draw"]).item()

    # Set up covariance function with mean hyperparameters
    cov_func = pm.gp.cov.ExpQuad(1, lengthscale_mean)
    mean_func = pm.gp.mean.Zero()
    
    # Create new model for prediction using standard Latent GP
    with pm.Model():
        gp_pred = pm.gp.Latent(cov_func=cov_func, mean_func=mean_func)
        
        # Conditional prediction
        mu_pred = gp_pred.conditional(
            "mu_pred",
            X_plot_scaled.reshape(-1, 1),
            given={
                "X": Xs.reshape(-1, 1),
                "f": y_train,  # Use observed values as function values
                "gp": gp_pred
            }
        )
        mu_pred_eval = pm.draw(mu_pred, 1).flatten()

    return mu_pred_eval


def analyze_reaction_time_data(csv_file, censor_limit=3.1, n_inducing=10, debug=False):
    """
    Complete analysis pipeline for reaction time data.
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file with reaction time data
    censor_limit : float
        Censoring threshold
    n_inducing : int
        Number of inducing points for sparse GP
    debug : bool
        Whether to print debug information
        
    Returns:
    --------
    tuple : (results_df, failures_df)
        DataFrames with successful fits and failures
    """
    # --- Data Preprocessing ---
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find file {csv_file}")
        print("Creating synthetic data for demonstration...")
        df = create_synthetic_data()
    
    selected_participants = df['participant'].unique()
    df = df[df['participant'].isin(selected_participants)].copy()
    df = df.sort_values(['participant', 'spawnrate']).reset_index(drop=True)
    df['participant_id'] = df['participant'].astype('category').cat.codes
    df['censored'] = (df['RT'] >= censor_limit).astype(int)
    df['RT_censored'] = np.where(df['censored'] == 1, censor_limit, df['RT'])

    # --- Loop over participants ---
    results = []
    failures = []

    for p in df['participant_id'].unique():
        print(f"\n==== Fitting participant {p} (sparse GP) ====")
        df_p = df[df['participant_id'] == p]
        X = df_p['spawnrate'].values
        y = df_p['RT_censored'].values

        print(f"  [participant DEBUG] Number of data points: {len(X)}")
        print(f"  [participant DEBUG] Censored counts: {np.bincount(df_p['censored'])}")
        print(f"  [participant DEBUG] X shape: {X.shape}")
        print(f"  [participant DEBUG] y shape: {y.shape}")

        # Skip participants with insufficient data
        if len(np.unique(y)) < 3:
            print(f"  Participant {p}: Not enough variation in RT_censored, skipping.")
            failures.append({"participant_id": p, "reason": "not enough variation"})
            continue
        if np.all(y >= censor_limit):
            print(f"  Participant {p}: All RTs are censored, skipping.")
            failures.append({"participant_id": p, "reason": "all censored"})
            continue

        # Fit the model
        trace, scaler, Xu, error, Xs, y_train = fit_participant_sparse_gp(
            df_p, n_inducing=n_inducing, censor_limit=censor_limit, debug=debug
        )
        
        if trace is None:
            print(f"  Participant {p}: Fitting failed with error: {error}")
            failures.append({"participant_id": p, "reason": "model fit error", "error": str(error)})
            continue

        # Make predictions
        X_med, X_scale = scaler
        X_plot = np.linspace(X.min(), X.max(), 100)
        X_plot_scaled = (X_plot - X_med) / X_scale

        print("  [prediction DEBUG] X_plot_scaled shape:", X_plot_scaled.shape)
        print("  [prediction DEBUG] Xs shape:", Xs.shape)
        print("  [prediction DEBUG] Xu shape:", Xu.shape)
        print("  [prediction DEBUG] y_train shape:", y_train.shape)

        # Get predictions
        y_pred_gp = predict_gp(trace, scaler, Xu, Xs, y_train, X_plot)
        print("  [prediction DEBUG] y_pred_gp shape:", y_pred_gp.shape)

        # Calculate RMSE
        rmse = np.sqrt(np.mean((y_train - np.interp(X, X_plot, y_pred_gp)) ** 2))

        # Extract model parameters
        lengthscale_mean = trace.posterior["lengthscale"].mean(dim=["chain", "draw"]).item()
        sigma_mean = trace.posterior["sigma"].mean(dim=["chain", "draw"]).item()

        results.append({
            "participant_id": p,
            "rmse": rmse,
            "lengthscale": lengthscale_mean,
            "sigma": sigma_mean,
            "n_inducing": len(Xu),
        })

        # Plot results
        cm = 1/2.54
        plt.figure(figsize=(10*cm, 8*cm))
        plt.scatter(X, y_train, label="Observed RT (censored)", alpha=0.7)
        plt.plot(X_plot, y_pred_gp, color="red", label=f"Sparse GP fit (inducing: {len(Xu)})")
        plt.axhline(censor_limit, linestyle="--", color="black", label="Censor limit")
        plt.xlabel("Spawnrate")
        plt.ylabel("Reaction Time")
        plt.legend()
        plt.title(f"Participant {p}: RMSE={rmse:.3f}, lengthscale={lengthscale_mean:.2f}")
        plt.grid(True)
        plt.show()

    features_df = pd.DataFrame(results)
    failures_df = pd.DataFrame(failures)
    
    print("\nSuccessful fits:")
    print(features_df)
    print("\nFailed fits:")
    print(failures_df)
    
    return features_df, failures_df


def create_synthetic_data():
    """Create synthetic reaction time data for testing."""
    np.random.seed(42)
    participants = ['P1', 'P2', 'P3']
    data = []
    
    for p in participants:
        n_obs = np.random.randint(20, 50)
        spawnrates = np.random.uniform(0.1, 2.0, n_obs)
        # Reaction time increases with spawn rate with some noise
        base_rt = 1.0 + 0.8 * spawnrates + np.random.normal(0, 0.3, n_obs)
        # Ensure positive reaction times
        base_rt = np.maximum(base_rt, 0.1)
        
        for i in range(n_obs):
            data.append({
                'participant': p,
                'spawnrate': spawnrates[i],
                'RT': base_rt[i]
            })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Run the analysis
    results_df, failures_df = analyze_reaction_time_data("nydata.csv", debug=True)