#!/usr/bin/env python3
import subprocess as sp
import time
import random
import logging
import argparse
from datetime import datetime

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for x in memory_free_info]
    return memory_free_values

def check_and_execute(command, gpu_requirement):
    try:
        while True:
            free_memory_values = get_gpu_memory()
            for gpu_index, free_memory in enumerate(free_memory_values):
                logging.debug(f"[{datetime.now()}] GPU {gpu_index}:")
                logging.debug(f"  Free memory: {free_memory / 1024:.2f} GB")

                if free_memory >= gpu_requirement * 1024:  # Convert GB to MB
                    logging.info(f"Sufficient memory available on GPU {gpu_index}. Executing command: {command}")
                    sp.run(command, shell=True)
                    return

            # Wait for 5 + random backoff minutes
            wait_time = 300 + random.randint(0, 300)
            logging.debug(f"[{datetime.now()}] Waiting for {wait_time // 60} minutes before checking again...")
            time.sleep(wait_time)
    except KeyboardInterrupt:
        logging.info("Exiting GPU Allocator without execution.")
    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check GPU memory and execute a command when conditions are met.")
    parser.add_argument("command", type=str, help="The command to execute when GPU memory requirements are met.")
    parser.add_argument("gpu_requirement", type=float, help="The minimum GPU memory required in GB.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    check_and_execute(args.command, args.gpu_requirement)
