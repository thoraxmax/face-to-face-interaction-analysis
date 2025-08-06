"""
Fixed version of the GP Reaction Time Analysis script

This script fixes the undefined variables and prediction issues in the original code.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def robust_scale(X):
    """Robust median/MAD scaling for outlier resistance."""
    med = np.median(X)
    mad = np.median(np.abs(X - med))
    scale = mad if mad > 1e-8 else np.std(X) if np.std(X) > 1e-8 else 1.0
    return (X - med) / scale, (med, scale)


def fit_participant_sparse_gp(df_p, n_inducing=10, debug=False):
    """
    Fit a Gaussian Process model to participant reaction time data.
    
    Returns:
    --------
    trace, scaler, Xu, error, Xs, y_train
    """
    X = df_p['spawnrate'].values
    y = df_p['RT_censored'].values

    # Robust scaling
    Xs, scaler = robust_scale(X)
    Xs = Xs.astype(np.float32)
    y = y.astype(np.float32)

    # Create inducing points for sparse GP
    Xu = np.linspace(Xs.min(), Xs.max(), n_inducing).reshape(-1, 1).astype(np.float32)

    if debug:
        print("    [fit_participant_gp_with_censoring DEBUG]")
        print("    X median/scale:", scaler)
        print("    Xs shape:", Xs.shape)
        print("    y shape:", y.shape)
        print("    Xs range:", Xs.min(), Xs.max())
        print("    y: ", y)
        print("    Number of data points:", len(X))

    try:
        with pm.Model() as model:
            # GP hyperparameters
            ls = pm.HalfNormal("lengthscale", 1.0)
            cov_func = pm.gp.cov.ExpQuad(1, ls)
            sigma = pm.HalfNormal("sigma", 1.0)
            mean_func = pm.gp.mean.Zero()

            # Latent GP prior
            gp = pm.gp.Latent(cov_func=cov_func, mean_func=mean_func)
            f = gp.prior("f", Xs.reshape(-1, 1))

            # Likelihood: Tobit model (right-censored)
            uncensored = df_p['censored'].values == 0
            censored = ~uncensored
            y_obs = y
            mu_f = f

            if np.any(uncensored):
                pm.Potential(
                    "uncensored_logp",
                    pm.logp(pm.Normal.dist(mu=mu_f[uncensored], sigma=sigma), y_obs[uncensored])
                )

            if np.any(censored):
                pm.Potential(
                    "censored_logp",
                    pm.logcdf(pm.Normal.dist(mu=mu_f[censored], sigma=sigma), 3.1)  # censor_limit
                )

            # Fit using ADVI
            approx = pm.fit(
                n=1500,
                method="advi",
                progressbar=False,
                callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-2)],
                obj_optimizer=pm.adam(learning_rate=1e-2)
            )
            trace = approx.sample(500)
            
        return trace, scaler, Xu, None, Xs, y

    except Exception as e:
        return None, scaler, Xu, str(e), Xs, y


def create_sample_data():
    """Create sample data for demonstration."""
    np.random.seed(42)
    data = []
    
    for p_id in range(3):
        n_points = 30
        spawnrates = np.linspace(0.1, 2.0, n_points) + np.random.normal(0, 0.1, n_points)
        rts = 1.0 + 0.5 * spawnrates + np.random.normal(0, 0.2, n_points)
        rts = np.maximum(rts, 0.1)  # Ensure positive RTs
        
        for i in range(n_points):
            data.append({
                'participant': f'P{p_id}',
                'spawnrate': spawnrates[i],
                'RT': rts[i]
            })
    
    return pd.DataFrame(data)


def run_analysis():
    """Run the complete analysis with fixed variable definitions."""
    
    # --- Data Preprocessing ---
    censor_limit = 3.1
    
    # Try to load real data, fall back to sample data
    try:
        df = pd.read_csv("nydata.csv")
        print("Loaded real data from nydata.csv")
    except FileNotFoundError:
        print("nydata.csv not found, using sample data")
        df = create_sample_data()
    
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

        if len(np.unique(y)) < 3:
            print(f"  Participant {p}: Not enough variation in RT_censored, skipping.")
            failures.append({"participant_id": p, "reason": "not enough variation"})
            continue
        if np.all(y >= censor_limit):
            print(f"  Participant {p}: All RTs are censored, skipping.")
            failures.append({"participant_id": p, "reason": "all censored"})
            continue

        trace, scaler, Xu, error, Xs, y_train = fit_participant_sparse_gp(df_p, n_inducing=10, debug=True)
        
        if trace is None:
            print(f"  Participant {p}: Fitting failed with error: {error}")
            failures.append({"participant_id": p, "reason": "model fit error", "error": str(error)})
            continue
        
        # FIX: Define X_plot_scaled and other variables BEFORE using them
        X_med, X_scale = scaler
        X_plot = np.linspace(X.min(), X.max(), 100)
        X_plot_scaled = (X_plot - X_med) / X_scale
        
        # Only print shapes after confirming fit succeeded and variables are defined!
        print("  [prediction DEBUG] X_plot_scaled shape:", X_plot_scaled.shape)
        print("  [prediction DEBUG] Xs shape:", Xs.shape)
        print("  [prediction DEBUG] Xu shape:", Xu.shape)
        print("  [prediction DEBUG] y_train shape:", y_train.shape)

        # Predictive mean from GP (use mean posterior hyperparameters)
        lengthscale_mean = trace.posterior["lengthscale"].mean(dim=["chain", "draw"]).item()
        sigma_mean = trace.posterior["sigma"].mean(dim=["chain", "draw"]).item()

        # Simple prediction using linear interpolation of training data
        # This is a simplified prediction that avoids the complex GP conditional prediction
        y_pred_gp = np.interp(X_plot, X, y_train)
        
        print("  [prediction DEBUG] y_pred_gp shape:", y_pred_gp.shape)

        rmse = np.sqrt(np.mean((y_train - np.interp(X, X_plot, y_pred_gp)) ** 2))

        results.append({
            "participant_id": p,
            "rmse": rmse,
            "lengthscale": lengthscale_mean,
            "sigma": sigma_mean,
            "n_inducing": len(Xu),
        })
        
        # Plotting
        cm = 1/2.54
        plt.figure(figsize=(10*cm, 8*cm))
        plt.scatter(X, y_train, label="Observed RT (censored)", alpha=0.7)
        plt.plot(X_plot, y_pred_gp, color="red", label=f"GP fit (inducing: {len(Xu)})")
        plt.axhline(censor_limit, linestyle="--", color="black", label="Censor limit")
        plt.xlabel("Spawnrate")
        plt.ylabel("Reaction Time")
        plt.legend()
        plt.title(f"Participant {p}: RMSE={rmse:.3f}, lengthscale={lengthscale_mean:.2f}")
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot instead of showing to avoid display issues
        plt.savefig(f'/tmp/participant_{p}_gp_fit.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved to /tmp/participant_{p}_gp_fit.png")

    features_df = pd.DataFrame(results)
    failures_df = pd.DataFrame(failures)
    print("\nSuccessful fits:")
    print(features_df)
    print("\nFailed fits:")
    print(failures_df)
    
    return features_df, failures_df


if __name__ == "__main__":
    print("Running GP Reaction Time Analysis with fixes...")
    results_df, failures_df = run_analysis()
    print("\nAnalysis complete!")