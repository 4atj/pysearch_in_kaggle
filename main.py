__all__ = (
    "push_kernel",
    "get_kernel_status",
    "save_kernel_output",
    "main",
)

import argparse
import asyncio
import datetime
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile

from dotenv import dotenv_values


async def push_kernel(
    kernel_metadata_dir: str,
    env: dict[str, str],
    delay: int = 15,
) -> str:
    while True:
        response = subprocess.run(
            ["kaggle", "kernels", "push", "-p", kernel_metadata_dir],
            env=env,
            capture_output=True,
        )

        stdout = response.stdout.decode()
        stderr = response.stderr.decode()

        if "push error: Maximum batch CPU" in stdout or "MaxRetryError" in stderr:
            await asyncio.sleep(delay)
            continue

        _, username, kernel_slug = stdout.strip().rsplit("/", 2)
        kernel_ref = f"{username}/{kernel_slug}"

        return kernel_ref


async def get_kernel_status(
    kernel_ref: str, env: dict[str, str], delay: int = 15
) -> str:
    while True:
        response = subprocess.run(
            ["kaggle", "kernels", "status", kernel_ref],
            env=env,
            capture_output=True,
        )

        stdout = response.stdout.decode()
        stderr = response.stderr.decode()

        if "MaxRetryError" in stderr:
            await asyncio.sleep(delay)
            continue

        return stdout.split('"')[1]


async def save_kernel_output(
    output_dir: str, kernel_ref: str, env: dict[str, str], delay: int = 15
):
    while True:
        response = subprocess.run(
            ["kaggle", "kernels", "output", kernel_ref, "-p", output_dir],
            env=env,
            capture_output=True,
        )

        stderr = response.stderr.decode()

        if "MaxRetryError" in stderr:
            await asyncio.sleep(delay)
            continue

        _username, kernel_slug = kernel_ref.split("/")

        if not os.path.exists(os.path.join(output_dir, f"{kernel_slug}.log")):
            raise RuntimeError(f"Failed to download the log of kernel {kernel_ref}.")

        return


async def pysearch(
    kernel_metadata_dir: str,
    kernel_title: str,
    env: dict[str, str],
    logger: logging.Logger = logging.root,
    chunk_id: int = 0,
    number_of_chunks: int = 1,
) -> tuple[int, str, str]:
    assert chunk_id < number_of_chunks

    with tempfile.TemporaryDirectory() as kernel_metadata_dir_copy:
        for item in os.listdir(kernel_metadata_dir):
            source = os.path.join(kernel_metadata_dir, item)
            destination = os.path.join(kernel_metadata_dir_copy, item)
            if os.path.isfile(source):
                shutil.copy(source, destination)
            elif os.path.isdir(source):
                shutil.copytree(source, destination)

        kernel_metadata_path = os.path.join(
            kernel_metadata_dir_copy, "kernel-metadata.json"
        )

        with open(kernel_metadata_path, "r") as fp:
            kernel_metadata = json.load(fp)

        kernel_metadata["title"] = kernel_title

        with open(kernel_metadata_path, "w") as fp:
            json.dump(kernel_metadata, fp)

        code_path = os.path.join(kernel_metadata_dir_copy, kernel_metadata["code_file"])

        with open(code_path, "r") as fp:
            code = fp.read()

        code = code.replace(
            "NUMBER_OF_CHUNKS: usize = 1",
            f"NUMBER_OF_CHUNKS: usize = {number_of_chunks}",
        ).replace("CHUNK_ID: usize = 0", f"CHUNK_ID: usize = {chunk_id}")

        with open(code_path, "w") as fp:
            fp.write(code)

        kernel_ref = await push_kernel(kernel_metadata_dir_copy, env)

    logger.info(f"Started kernel {kernel_ref}")

    with tempfile.TemporaryDirectory() as output_temp_dir:
        status = ""
        while status != "complete":
            new_status = await get_kernel_status(
                kernel_ref,
                env,
            )

            if new_status != status:
                status = new_status
                logger.info(f"Kernel {kernel_ref} status: {status}")

            if status == "error":
                raise RuntimeError(f"Kernel {kernel_ref} failed")

            await asyncio.sleep(15)

        await save_kernel_output(output_temp_dir, kernel_ref, env)

        _username, kernel_slug = kernel_ref.split("/")

        with open(os.path.join(output_temp_dir, f"{kernel_slug}.log")) as fp:
            stdout = "".join(
                event["data"]
                for event in json.load(fp)
                if event["stream_name"] == "stdout"
            )

    return chunk_id, kernel_ref, stdout


def chunked_pysearch(
    kernel_metadata_dir: str,
    env: dict[str, str],
    logger: logging.Logger = logging.root,
    number_of_chunks: int = 1,
    start_chunk_id: int = 0,
    end_chunk_id: int | None = None,
) -> list[asyncio.Task[tuple[int, str, str]]]:
    if end_chunk_id is None:
        end_chunk_id = number_of_chunks

    assert 0 <= start_chunk_id < end_chunk_id <= number_of_chunks

    start_datetime = datetime.datetime.now()

    tasks: list[asyncio.Task[tuple[int, str, str]]] = []

    for chunk_id in range(start_chunk_id, end_chunk_id):
        tasks.append(
            asyncio.create_task(
                pysearch(
                    kernel_metadata_dir,
                    f"{start_datetime}-chunk-{chunk_id}-{number_of_chunks}",
                    env,
                    logger,
                    chunk_id,
                    number_of_chunks,
                )
            )
        )

    return tasks


def _create_logger(name: str, dir_path: str) -> logging.Logger:
    formatter = logging.Formatter(
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        fmt="{asctime}: {message}",
    )
    handler = logging.FileHandler(os.path.join(dir_path, f"{name}.log"), mode="w")
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger


async def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--path",
        type=str,
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "pysearch"),
        help="The kernel metadata dir path",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=1,
        help="The number of chunks the work is split into",
    )
    parser.add_argument("--start", type=int, default=0, help="Start chunk id")
    parser.add_argument(
        "--end", type=int, default=None, help="End chunk id (exclusive)"
    )
    args = parser.parse_args()

    path: str = args.path
    chunks: int = args.chunks
    start: int = args.start
    end: int | None = args.end

    output_dir_path = os.path.join(path, "results")
    os.makedirs(output_dir_path, exist_ok=True)

    logger = _create_logger("pysearch", output_dir_path)

    env = os.environ | {k: v or "" for k, v in dotenv_values().items()}

    with open(os.path.join(output_dir_path, "stdout"), "w") as fp:
        for result in asyncio.as_completed(
            chunked_pysearch(path, env, logger, chunks, start, end),
        ):
            chunk_id, kernel_ref, stdout = await result
            fp.write(f"Chunk: {chunk_id} Kernel: {kernel_ref}\n{stdout}")
            fp.flush()


if __name__ == "__main__":
    asyncio.run(main())
