import subprocess
import time
import sys
import logging

from _utils.config import get_env

LOG_PATH = get_env("MAIN_LOG_PATH")

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

processes = []

def start_process(module: str) -> None:
    """Start a Python module in a new subprocess."""

    p = subprocess.Popen([sys.executable, "-m", module])
    processes.append(p)
    logging.info(f"Started {module} (pid {p.pid})")
    print(f"Started {module} (pid {p.pid})")

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
        start_process("prod_workflow.data_manager")
        time.sleep(2)
        start_process("prod_workflow.inference_loop")
        time.sleep(2)
        start_process("prod_workflow.meta_inference_loop")
        time.sleep(2)
        start_process("prod_workflow.brain")
        time.sleep(2)
        # start_process("prod_workflow.broker")

        logging.info("All processes started successfully.")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        stop_all()
    except Exception as e:
        logging.error(f"Main controller error: {e}", exc_info=True)
        stop_all()
