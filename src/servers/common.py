import signal
import sys


GLOBAL_PROCESS_LIST = []


def kill_servers():
    """Kill all processes."""
    for p in GLOBAL_PROCESS_LIST:
        p.terminate()
        p.join()


def graceful_exit(sig, frame):
    """Kill all processes on SIGINT."""
    _ = sig, frame  # Unused
    kill_servers()
    sys.exit()


signal.signal(signal.SIGINT, graceful_exit)
