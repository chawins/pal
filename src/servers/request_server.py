import logging
import multiprocessing as mp
import os
import time
from typing import Any

import requests
from tqdm import tqdm

from src.servers.common import GLOBAL_PROCESS_LIST, kill_servers

_BASE_SLEEP_TIME = 3
REQUEST_ERRORS = (
    requests.exceptions.HTTPError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.RequestException,
    requests.exceptions.JSONDecodeError,
)

Completions = list[dict[str, Any]]
Queues = tuple[mp.Queue, mp.Queue]

logger = logging.getLogger(__name__)


def _request_server(call_queue, leader=False):
    while True:
        task = call_queue.get(block=True)
        if task is None:
            return

        compl_id, message, url, headers, kwargs, dest_queue = task
        result = send_request(message, url=url, headers=headers, **kwargs)
        if result == 0 and not leader:
            call_queue.put(task)
            print("Reducing the number of threads due to Rate Limit")
            return
        if result == 0 and leader:
            call_queue.put(task)
        else:
            dest_queue.put((compl_id, result))


def send_request(
    messages: str,
    url: str = "https://api.together.xyz/v1/completions",
    headers: dict[str, str] | None = None,
    **kwargs,
):
    def loop(params):
        max_retries = 7
        retry = 0
        while retry < max_retries:
            try:
                return requests.post(
                    url,
                    json=kwargs,
                    headers=headers,
                    timeout=_BASE_SLEEP_TIME * (1 + retry),
                )
            except REQUEST_ERRORS as e:
                if retry == max_retries - 1:
                    print(f"Error after {retry} retries: {e}\n{params}")
                    if "This model does not support the 'logprobs'" in str(e):
                        print("This is a rare client-side error.")
                    print("Returning None response and skipping...")
                    return None
                if "maximum context length" in str(e):
                    print("Context length exceeded")
                    raise e

                print(e)
                time.sleep(_BASE_SLEEP_TIME * (1 + retry))
                retry += 1

    assert isinstance(messages, str), messages
    kwargs["prompt"] = messages
    return loop(kwargs)


def init_servers(number_of_processes: int = 4):
    """Initializes multiple chat servers using mp.

    Args:
        number_of_processes (int): The number of server processes to start.
            Default is 4.

    Returns:
        tuple: A tuple containing a call queue and a global manager object.
    """
    global_manager = mp.Manager()
    call_queue = global_manager.Queue()

    for i in range(number_of_processes):
        p = mp.Process(target=_request_server, args=(call_queue, i == 0))
        p.daemon = True
        p.start()
        GLOBAL_PROCESS_LIST.append(p)

    return call_queue, global_manager


def standalone_server(
    inputs: list[str],
    display_progress: bool = True,
    num_processes: int = os.cpu_count(),
    cleanup: bool = True,
    queues: Queues | None = None,
    url: str = "https://api.together.xyz/v1/completions",
    headers: dict[str, str] | None = None,
    **kwargs,
) -> Completions | tuple[Completions, Queues]:
    """Run a standalone server to process inputs and return responses.

    Args:
        inputs: A string or a list of strings representing the inputs to be
            processed.
        **kwargs: Additional keyword arguments for server configuration.

    Returns:
        If `inputs` is a string, returns a single response string.
        If `inputs` is a list of strings, returns a list of response strings.
    """
    if queues is None:
        logger.debug("Starting server with %d processes", num_processes)
        queue, mgr = init_servers(number_of_processes=num_processes)
        resp_queue = mgr.Queue()
    else:
        logger.debug("Not starting new server. Using existing queues.")
        queue, resp_queue = queues

    assert isinstance(inputs, list) and isinstance(
        inputs[0], (str, list)
    ), inputs
    logger.debug("Calling API with following params: %s", kwargs)

    for idx, inpt in enumerate(inputs):
        queue.put((idx, inpt, url, headers, kwargs, resp_queue))

    responses = ["" for _ in inputs]
    for _ in tqdm(
        inputs,
        total=len(inputs),
        desc="Querying API",
        disable=not display_progress,
    ):
        idx, resp = resp_queue.get(block=True)
        responses[idx] = resp

    if cleanup:
        kill_servers()
        return responses
    return responses, (queue, resp_queue)
