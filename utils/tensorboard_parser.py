#!/usr/bin/env python
import os
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd


def parse_tensorboard_logs(log_dir):
    """
    Parse TensorBoard logs to extract metrics from multiple experiments.

    Args:
        log_dir (str): Directory containing TensorBoard log subdirectories.

    Returns:
        pd.DataFrame: A DataFrame with columns ['experiment', 'step', 'metric_name', 'value'].
    """
    metrics = []

    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if "events.out.tfevents" in file:
                print(f"Processing: {root}/{file}")
                experiment_name = os.path.relpath(root, log_dir)  # Relative path as experiment name
                event_path = os.path.join(root, file)
                event_acc = EventAccumulator(event_path)
                event_acc.Reload()

                # Extract scalar metrics
                for tag in event_acc.Tags()['scalars']:
                    for scalar_event in event_acc.Scalars(tag):
                        metrics.append({
                            "experiment": experiment_name,
                            "step": scalar_event.step,
                            "metric_name": tag,
                            "value": scalar_event.value
                        })

    return pd.DataFrame(metrics)


def main():
    parser = argparse.ArgumentParser(description="Parse TensorBoard logs and save metrics to a CSV file.")
    parser.add_argument("--log-dir", type=str, required=True, help="Path to the directory containing TensorBoard logs.")
    parser.add_argument("--output-csv", type=str, required=True, help="Path to save the extracted metrics as a CSV file.")
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        raise FileNotFoundError(f"The specified log directory does not exist: {args.log_dir}")

    metrics_df = parse_tensorboard_logs(args.log_dir)

    if metrics_df.empty:
        print("No metrics found in the specified log directory.")
        return

    metrics_df.to_csv(args.output_csv, index=False)
    print(f"Metrics saved to {args.output_csv}")


if __name__ == "__main__":
    main()
