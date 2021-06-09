import os
from glob import glob

import pandas as pd
import wandb

from efficiency.evaluation import compute_metrics

api = wandb.Api()

if not os.path.exists("efficiency/analysis/wandb.csv"):
    runs = api.runs("tud-cs4245-group-27/Compressing StyleGAN")

    run_info = []
    for run in runs:
        run_info.append(
            {
                **run.summary._json_dict,
                **{k: v for k, v in run.config.items() if not k.startswith("_")},
                **{"name": run.name, "id": run.id},
            }
        )

    df = pd.DataFrame.from_records(run_info)
    df.to_csv("efficiency/analysis/wandb.csv")
else:
    df = pd.read_csv("efficiency/analysis/wandb.csv")
print(df)

for run_id, run_dir in zip(df.id, df.run_dir):
    print(list(sorted(glob(run_dir + "/*.pkl")))[-1])
    most_recent_pkl = list(sorted(glob(run_dir + "/*.pkl")))[-1]
    metrics = compute_metrics(most_recent_pkl, datapath="/home/hans/trainsets/ffhq256.zip")

    run = api.run(f"tud-cs4245-group-27/Compressing StyleGAN/{run_id}")

    for name, value in metrics.items():
        run.summary[f"Post Evaluation/{name.upper() if 'id' in name else name.capitalize()}"] = value
    run.summary.update()
    exit(0)
