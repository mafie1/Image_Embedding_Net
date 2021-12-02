import optuna
from predict_mask import apply_on_val_set, cluster_single_image

def objective(trial):
    n = trial.suggest_float('n', 10, 500)
    DiC, SBD = apply_on_val_set(n_min=n, epsilon=0.5, method='hdbscan')

    return SBD

study = optuna.create_study(direction='maximize')

study.optimize(objective, n_trials=2)

#print(study.best_params)

"""
def training_function(config):
    # Hyperparameters
    alpha, beta = config["alpha"], config["beta"]
    for step in range(10):
        # Iterative training function - can be any arbitrary training procedure.
        intermediate_score = objective(step, alpha, beta)
        # Feed the score back back to Tune.
        tune.report(mean_loss=intermediate_score)


analysis = tune.run(
    training_function,
    config={
        "alpha": tune.grid_search([0.001, 0.01, 0.1]),
        "beta": tune.choice([1, 2, 3])
    })

print("Best config: ", analysis.get_best_config(
    metric="mean_loss", mode="min"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df
"""