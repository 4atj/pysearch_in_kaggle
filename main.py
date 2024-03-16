import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import tempfile
import time

from dotenv import dotenv_values


class KaggleError(Exception):
    pass


class KaggleLimitError(KaggleError):
    pass


class KaggleConnectionError(KaggleError, ConnectionError):
    pass


class KaggleAlreadyPushedError(KaggleError):
    pass


class KaggleKernel:
    def __init__(
        self, save_path: str, env: dict[str, str], include_current_env: bool = True
    ):
        self.ref = ""
        self.status = "not_pushed"
        self.local_env = env
        self.include_current_env = include_current_env
        self.save_path = save_path

    @property
    def env(self) -> dict[str, str]:
        if self.include_current_env:
            return os.environ | self.local_env
        return self.local_env

    @property
    def pushed(self) -> bool:
        return self.status != "not_pushed"

    @property
    def finished(self) -> bool:
        return self.status in ("complete", "error", "cancelAcknowledged")

    def push(self, kernel_metadata_dir_path: str):
        if self.pushed:
            raise KaggleAlreadyPushedError

        try:
            response = subprocess.run(
                ["kaggle", "kernels", "push", "-p", kernel_metadata_dir_path],
                env=self.env,
                check=True,
                capture_output=True,
                encoding="utf-8",
            )
            if "push error: Maximum batch CPU" in response.stdout:
                raise KaggleLimitError
        except subprocess.CalledProcessError as exception:
            print(exception.stdout)
            if "is already in use" in exception.stdout:
                raise KaggleAlreadyPushedError from exception
            if "MaxRetryError" in exception.stderr:
                raise KaggleConnectionError from exception
            raise KaggleError from exception

        *_, username, kernel_slug = response.stdout.strip().split("/")
        self.ref = f"{username}/{kernel_slug}"
        self.status = "pending"

    def update_status(self):
        try:
            response = subprocess.run(
                ["kaggle", "kernels", "status", self.ref],
                env=self.env,
                check=True,
                capture_output=True,
                encoding="utf-8",
            )
        except subprocess.CalledProcessError as exception:
            if "MaxRetryError" in exception.stderr:
                raise KaggleConnectionError from exception
            raise KaggleError from exception

        self.status = response.stdout.split('"')[1]

    def save_output(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        try:
            subprocess.run(
                ["kaggle", "kernels", "output", self.ref, "-p", self.save_path],
                env=self.env,
                check=True,
                capture_output=True,
                encoding="utf-8",
            )
        except subprocess.CalledProcessError as exception:
            if "MaxRetryError" in exception.stderr:
                raise KaggleConnectionError from exception
            raise KaggleError from exception


def pysearch_push(
    kernel: KaggleKernel,
    run_path: str,
    chunk_id: int = 0,
    number_of_chunks: int = 1,
):
    assert 0 <= chunk_id < number_of_chunks

    kernel_metadata_dir_path = os.path.join(run_path, "pysearch")

    with tempfile.TemporaryDirectory() as temp_kernel_metadata_dir_path:
        for filename in os.listdir(kernel_metadata_dir_path):
            source_file = os.path.join(kernel_metadata_dir_path, filename)
            dest_file = os.path.join(temp_kernel_metadata_dir_path, filename)
            shutil.copy(source_file, dest_file)

        kernel_metadata_dir_path = os.path.join(
            temp_kernel_metadata_dir_path, "kernel-metadata.json"
        )
        with open(kernel_metadata_dir_path, "r") as fp:
            kernel_metadata = json.load(fp)
        with open(kernel_metadata_dir_path, "w") as fp:
            title = (
                f"{os.path.split(run_path)[-1]}_{chunk_id}_{datetime.datetime.now()}"
            )
            slug = re.sub(r"\W+", "-", title).lower().strip("-")
            kernel_metadata["title"] = title
            kernel_metadata["id"] = f"{{username}}/{slug}"
            json.dump(kernel_metadata, fp)

        code_path = os.path.join(
            temp_kernel_metadata_dir_path, kernel_metadata["code_file"]
        )
        with open(code_path, "r") as fp:
            code = fp.read()
        with open(code_path, "w") as fp:
            code = code.replace(
                "NUMBER_OF_CHUNKS: usize = 1",
                f"NUMBER_OF_CHUNKS: usize = {number_of_chunks}",
            ).replace("CHUNK_ID: usize = 0", f"CHUNK_ID: usize = {chunk_id}")
            fp.write(code)

        kernel.push(temp_kernel_metadata_dir_path)


def save_pysearch_kernels(
    run_path: str, number_of_chunks: int, kernels: dict[int, KaggleKernel]
):
    with open(os.path.join(run_path, "kernels.json"), "w") as fp:
        json.dump(
            {
                "number_of_chunks": number_of_chunks,
                "kernels": {
                    chunk_id: kernel.__dict__
                    | {"save_path": os.path.relpath(kernel.save_path, run_path)}
                    for chunk_id, kernel in kernels.items()
                },
            },
            fp,
        )
    with open(os.path.join(run_path, "results"), "w") as fp:
        for chunk_id, kernel in kernels.items():
            if kernel.finished:
                fp.write(f"Chunk {chunk_id}\n")
                try:
                    with open(
                        os.path.join(kernel.save_path, "results"), "r"
                    ) as chunk_fp:
                        fp.write(chunk_fp.read())
                except FileNotFoundError:
                    fp.write("no results file found")
                fp.write("\n")


def load_pysearch_kernels(run_path: str) -> tuple[int, dict[int, KaggleKernel]]:
    from typing import Any, cast

    with open(os.path.join(run_path, "kernels.json"), "r") as fp:
        data: dict[str, Any] = json.load(fp)
        kernels: dict[int, KaggleKernel] = {}
        for chunk_id, kernel_data in cast(
            dict[str, dict[str, object]], data["kernels"]
        ).items():
            kernel = KaggleKernel("", {})
            for k, v in kernel_data.items():
                setattr(kernel, k, v)
            kernel.save_path = os.path.join(run_path, kernel.save_path)
            kernels[int(chunk_id)] = kernel
        return int(data["number_of_chunks"]), kernels


def main():
    my_path = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(
        description="Run pysearch in multiple kaggle kernel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command")

    start_parser = subparsers.add_parser(
        "start",
        description="Start a new pysearch run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    start_parser.add_argument(
        "--kernel_path",
        type=str,
        default=os.path.join(my_path, "pysearch"),
        help="The path to the directory containing the kernel metadata.",
    )
    
    start_parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="The path to the directory where the run data will be stored.",
    )

    start_parser.add_argument(
        "--chunks",
        type=int,
        default=1,
        help="The number of chunks the work is split into.",
    )

    start_parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First chunk id in the range of ids of chunks to run."
    )

    start_parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End chunk id (exclusive), None is equivalant to the last chunk.",
    )

    load_parser = subparsers.add_parser(
        "continue",
        description="Contiune a previous run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    load_parser.add_argument(
        "path",
        type=str,
        help="The path to the directory where the run data is stored.",
    )

    args = parser.parse_args()

    match args.command:
        case "start":
            if args.save_path is None:
                count = 0
                while True:
                    run_path = os.path.join(my_path, "runs", f"run_{count:03d}")
                    if not os.path.exists(run_path):
                        break
                    count += 1
            else:
                run_path: str = os.path.abspath(args.save_path)

            os.makedirs(run_path)

            shutil.copytree(args.kernel_path, os.path.join(run_path, "pysearch"))

            env = {k: v or "" for k, v in dotenv_values().items()}

            number_of_chunks: int = args.chunks
            kernels: dict[int, KaggleKernel] = {}

            start: int = args.start
            end: int = args.chunks if args.end is None else args.end

            if not 0 <= start < end <= number_of_chunks:
                raise ValueError(f"--start, --end values {start}, {end} are inappropriate")

            kernels = {
                chunk_id: KaggleKernel(
                    os.path.join(run_path, "kernel_outputs", f"chunk_{chunk_id:03d}"),
                    env,
                )
                for chunk_id in range(start, end)
            }

            save_pysearch_kernels(run_path, number_of_chunks, kernels)

        case "continue":
            run_path: str = os.path.abspath(args.path)
            number_of_chunks, kernels = load_pysearch_kernels(run_path)

        case _:
            raise ValueError("Invalid command")

    print(f"run_path: {run_path}")
    print(
        f"Be cautious your kaggle tokens are stored in {os.path.join('$run_path','kernels.json')}"
    )

    sleep_amount: int = 30
    while any(not k.finished for k in kernels.values()):
        for chunk_id, kernel in kernels.items():
            if kernel.finished:
                continue

            old_status = kernel.status

            if not kernel.pushed:
                try:
                    pysearch_push(kernel, run_path, chunk_id, number_of_chunks)
                    time.sleep(sleep_amount)
                except KaggleLimitError:
                    pass

            kernel.update_status()
            time.sleep(sleep_amount)

            if kernel.status != old_status:
                print(f"Chunk {chunk_id} status: {kernel.status}")
                if kernel.finished:
                    kernel.save_output()
                    time.sleep(sleep_amount)
                save_pysearch_kernels(run_path, number_of_chunks, kernels)


if __name__ == "__main__":
    main()
