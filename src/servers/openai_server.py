import logging
import multiprocessing as mp
import os
import time
from typing import Literal

import openai
from openai import OpenAI
from tqdm import tqdm

from src.servers.common import GLOBAL_PROCESS_LIST, kill_servers

_BASE_SLEEP_TIME = 3

ChatMessages = list[dict[str, str]]
Completions = list[openai.types.Completion]
Queues = tuple[mp.Queue, mp.Queue]

logger = logging.getLogger(__name__)


def openai_chat_server(call_queue, leader=False):
    """A function that listens to a call queue for incoming tasks, and processes them using OpenAI's API.

    Args:
        call_queue (Queue): A queue object that contains incoming tasks. These are made of the following elements:
        id: id for this task.
        message: a string representing the user's message prompt.
        max_tokens: an integer representing the maximum number of tokens to generate.
        kwargs: a dictionary containing optional keyword arguments to be passed to the call_openai function.
        dest_queue: a queue object where the result of the task will be put.

    Returns:
        None
    """
    client = OpenAI()

    while True:
        task = call_queue.get(block=True)
        if task is None:
            return

        compl_id, message, logit_bias, kwargs, dest_queue = task
        rslt = call_openai(client, message, logit_bias, **kwargs)
        if rslt == 0 and not leader:
            call_queue.put(task)
            print("Reducing the number of OpenAI threads due to Rate Limit")
            return
        if rslt == 0 and leader:
            call_queue.put(task)
        else:
            dest_queue.put((compl_id, rslt))


def call_openai(
    client: openai.Client,
    messages: str | ChatMessages,
    logit_bias: dict[int, float] | None,
    query_type: Literal["chat", "completion"] = "chat",
    **kwargs,
):
    """Calls the OpenAI API to generate text based on the given parameters.

    Args:
        client (openai.api_client.Client): The OpenAI API client.
        message (str): The user's message prompt.
        logit_bias: A dictionary containing logit bias values for each token.
        query_type (str): The type of completion to use. Defaults to "chat".

    Returns:
        The generated responses from the OpenAI API.
    """

    def loop(f, params):
        max_retries = 7
        retry = 0
        while retry < max_retries:
            try:
                return f(params)
            except (
                openai.InternalServerError,
                openai.APITimeoutError,
                openai.BadRequestError,
            ) as e:
                # Exceptions to not retry: AuthenticationError, NotFoundError,
                # TypeError
                if retry == max_retries - 1:
                    print(f"Error after {retry} retries: {e}\n{params}")
                    if "This model does not support the 'logprobs'" in str(e):
                        print("This is a rare client-side error.")
                    print("Returning None response and skipping...")
                    return None
                if "maximum context length" in str(e):
                    print("Context length exceeded")
                    raise e

                if "timed out" in str(e) and retry < 3:
                    params["timeout"] += _BASE_SLEEP_TIME * retry
                time.sleep(_BASE_SLEEP_TIME * (1 + retry))
                retry += 1

    kwargs["logit_bias"] = logit_bias

    if query_type == "chat":
        assert isinstance(messages, list) and isinstance(
            messages[0], dict
        ), messages
        kwargs["messages"] = messages
        return loop(lambda x: client.chat.completions.create(**x), kwargs)

    assert isinstance(messages, str), messages
    kwargs["prompt"] = messages
    return loop(lambda x: client.completions.create(**x), kwargs)


def init_servers(number_of_processes: int = 4):
    """Initializes multiple chat servers using mp.

    Args:
        number_of_processes (int): The number of server processes to start. Default is 4.

    Returns:
        tuple: A tuple containing a call queue and a global manager object.
    """
    global_manager = mp.Manager()
    call_queue = global_manager.Queue()

    for i in range(number_of_processes):
        p = mp.Process(target=openai_chat_server, args=(call_queue, i == 0))
        p.daemon = True
        p.start()
        GLOBAL_PROCESS_LIST.append(p)

    return call_queue, global_manager


def standalone_server(
    inputs: list[str | ChatMessages],
    logit_biases: list[dict[int, float] | None],
    display_progress: bool = True,
    num_processes: int = os.cpu_count(),
    cleanup: bool = True,
    queues: Queues | None = None,
    **kwargs,
) -> Completions | tuple[Completions, Queues]:
    """Run a standalone server to process inputs and return responses.

    Args:
        inputs: A string or a list of strings representing the inputs to be processed.
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

    kwargs["timeout"] = 60
    assert isinstance(inputs, list) and isinstance(
        inputs[0], (str, list)
    ), inputs
    assert len(inputs) == len(logit_biases), (len(inputs), len(logit_biases))
    logger.debug("Calling OpenAI API with following params: %s", kwargs)

    for idx, (inpt, logit_bias) in enumerate(zip(inputs, logit_biases)):
        queue.put((idx, inpt, logit_bias, kwargs, resp_queue))

    responses = ["" for _ in inputs]
    for _ in tqdm(
        inputs,
        total=len(inputs),
        desc="Querying OpenAI",
        disable=not display_progress,
    ):
        idx, resp = resp_queue.get(block=True)
        responses[idx] = resp

    if cleanup:
        kill_servers()
        return responses
    return responses, (queue, resp_queue)
