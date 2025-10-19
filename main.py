import subprocess
import time
import signal
import sys
import logging
import os

LOG_PATH = "trading_bot.log"

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

processes = []

def start_process(script):
    p = subprocess.Popen(["python", script])
    processes.append(p)
    logging.info(f"Started {script} (pid {p.pid})")
    print(f"Started {script} (pid {p.pid})")

def stop_all():
    print("\nStopping all processes...")
    logging.info("Stopping all subprocesses...")
    for p in processes:
        p.terminate()
    for p in processes:
        p.wait()
    logging.info("All subprocesses stopped.")
    print("All processes stopped.")
    sys.exit(0)

if __name__ == "__main__":
    try:
        start_process("data_manager.py")
        time.sleep(2)
        start_process("inference_loop.py")
        time.sleep(2)
        start_process("meta_inference_loop.py")
        time.sleep(2)
        start_process("brain.py")
        time.sleep(2)
        start_process("broker.py")

        logging.info("All processes started successfully.")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        stop_all()
    except Exception as e:
        logging.error(f"Main controller error: {e}", exc_info=True)
        stop_all()
