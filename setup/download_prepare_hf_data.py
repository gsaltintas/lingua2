# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import gzip
import math
import os
import subprocess
import time

import boto3
import requests
from botocore.exceptions import ClientError
from huggingface_hub import snapshot_download, HfApi


def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)


def download_dataset(repo_id, local_dir, allow_patterns):
    print(f"Downloading dataset from {repo_id} into {local_dir}...")
    max_retries = 5
    retry_delay = 10  # seconds
    for attempt in range(max_retries):
        try:
            snapshot_download(
                repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                resume_download=True,
                max_workers=16,  # Don't hesitate to increase this number to lower the download time
            )
            break
        except requests.exceptions.ReadTimeout:
            if attempt < max_retries - 1:
                print(f"Timeout occurred. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise
    print(f"Dataset downloaded to {local_dir}")

def adapter_func(self, data: dict, path: str, id_in_file: int | str):
    """
    The default data adapter to adapt input data into the datatrove Document format

    Args:
        data: a dictionary with the "raw" representation of the data
        path: file path or source for this sample
        id_in_file: its id in this particular file or source

    Returns: a dictionary with text, id, media and metadata fields

    """
    metadata = data.pop("metadata", {})
    if isinstance(metadata, str):
        import json

        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            pass
    if not isinstance(metadata, dict):
        metadata = {"metadata": metadata}
    if "embeddings" in data:
        data.pop("embeddings", None)

    return {
        "text": data.pop(self.text_key, ""),
        "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
        "media": data.pop("media", []),
        "metadata": metadata | data,  # remaining data goes into metadata
    }
    
def parquet_to_jsonl(
    dataset, work_dir, src_dir, tgt_dir, ntasks=64, preserve_subsets=False
):
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import JsonlWriter

    adapter = None
    if "fineweb_2_hq" in dataset:
        adapter = adapter_func

    if preserve_subsets:
        subsets = os.listdir(f"{src_dir}")
        pipe = []
        for subset in subsets:
            # skip the non data directories
            if not os.path.isdir(f"{src_dir}/{subset}"):
                continue
            if subset in [".cache", ".git", ".github", "datatrove", "terashuf"]:
                continue
            print(f"Processing subset: {subset}")
            pipe = [
                ParquetReader(
                    f"{src_dir}/{subset}/",
                    file_progress=True,
                    doc_progress=True,
                    glob_pattern="**/*.parquet",
                    adapter=adapter,
                ),
                JsonlWriter(
                    tgt_dir,
                    output_filename=f"{dataset}.{subset}" + ".chunk.${rank}.jsonl",
                    compression=None,
                ),
            ]
            pipeline_exec = LocalPipelineExecutor(
                pipeline=pipe,
                tasks=ntasks,
                logging_dir=os.path.join(work_dir, "datatrove", subset),
            )
            pipeline_exec.run()
        return
    else:
        pipe = [
            ParquetReader(
                src_dir,
                file_progress=True,
                doc_progress=True,
                glob_pattern="**/*.parquet",
                adapter=adapter,
            ),
            JsonlWriter(
                tgt_dir,
                output_filename=dataset + ".chunk.${rank}.jsonl",
                compression=None,
            ),
        ]

    pipeline_exec = LocalPipelineExecutor(
        pipeline=pipe, tasks=ntasks, logging_dir=os.path.join(work_dir, "datatrove")
    )
    pipeline_exec.run()


def setup_terashuf(work_dir):
    terashuf_dir = os.path.join(work_dir, "terashuf")
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")
    print(terashuf_executable)
    if os.path.exists(terashuf_executable):
        print("terashuf executable already exists. Skipping setup.")
        return terashuf_dir

    print("Setting up terashuf...")
    run_command(f"git clone https://github.com/alexandres/terashuf {terashuf_dir}")
    run_command(f"make -C {terashuf_dir}")
    return terashuf_dir


def main(
    dataset,
    memory,
    data_dir,
    seed=42,
    nchunks=32,
    preserve_subsets=False,
    upload_to_hf=False,
    hf_path=None,
    max_file_size=None,
    skip_download=False,
    k_validation=10000,
):
    # Configuration
    repo_id = {
        "fineweb_edu": "HuggingFaceFW/fineweb-edu",
        "stack_edu": "common-pile/stackv2_edu_filtered",
        "fineweb_2": "HuggingFaceFW/fineweb-2",
        "fineweb_2_hq": "epfml/FineWeb2-HQ",
        "fineweb_edu_10bt": "HuggingFaceFW/fineweb-edu",
        "fineweb_edu_100bt": "HuggingFaceFW/fineweb-edu",
        "dclm_baseline_1.0": "mlfoundations/dclm-baseline-1.0",
        "dclm_baseline_1.0_10prct": "mlfoundations/dclm-baseline-1.0",
    }[dataset]
    src_dir = f"{data_dir}/{dataset}"
    out_dir = f"{src_dir}_shuffled"
    os.makedirs(out_dir, exist_ok=True)
    work_dir = src_dir  # Directory of this Python file
    if dataset not in ["fineweb_2_hq", "stack_edu"]:
        src_dir = f"{src_dir}/data"
    prefix = f"{dataset}.chunk."
    orig_extension = {
        "fineweb_edu": ".jsonl",
        "stack_edu": ".json.gz",
        "fineweb_2": ".jsonl",
        "fineweb_2_hq": ".jsonl",
        "fineweb_edu_10bt": ".jsonl",
        "fineweb_edu_100bt": ".jsonl",
        "dclm_baseline_1.0": ".jsonl.zst",
        "dclm_baseline_1.0_10prct": ".jsonl.zst",
    }[dataset]
    cat_command = {
        "fineweb_edu": "cat {}",
        "fineweb_2": "cat {}",
        "fineweb_2_hq": "cat {}",
        "fineweb_edu_10bt": "cat {}",
        "fineweb_edu_100bt": "cat {}",
        "stack_edu": "zcat {} && echo",
        "dclm_baseline_1.0": "zstdcat {} && echo",
        "dclm_baseline_1.0_10prct": "zstdcat {} && echo",
    }[dataset]
    allow_patterns = {
        "fineweb_edu": None,
        "fineweb_edu_10bt": "sample/10BT/*",
        "fineweb_edu_100bt": "sample/100BT/*",
        "fineweb_2": [
            "data/arb_Arab/*/000_0000[01].parquet",
            "data/ben_Beng/*/000_00000.parquet",
            "data/eng_Latn/*/000_0000[01].parquet",
            "data/deu_Latn/*/000_0000[01].parquet",
            "data/fra_Latn/*/000_0000[01].parquet",
            "data/ita_Latn/*/000_0000[01].parquet",
            "data/hin_Deva/*/000_00000.parquet",
            "data/jpn_Jpan/*/000_0000[01].parquet",
            "data/kor_Hang/*/000_0000[01].parquet",
            "data/rus_Cyrl/*/000_0000[012].parquet",
            "data/spa_Latn/*/000_0000[01].parquet",
            "data/swh_Latn/*/000_00000.parquet",
            "data/tel_Telu/*/000_00000.parquet",
            "data/tha_Thai/*/000_0000[01].parquet",
            "data/tur_Latn/*/000_0000[01].parquet",
            "data/cmn_Hani/*/000_0000[012].parquet",
        ],
        "fineweb_2_hq": [
            "ita_Latn/*",
            "tur_Latn/*",
            "fas_Arab/*",
            "cmn_Hani/*",
        ],
        "stack_edu": "*.json.gz",
        "dclm_baseline_1.0": "*.jsonl.zst",
        "dclm_baseline_1.0_10prct": "global-shard_01_of_10/*.jsonl.zst",
    }[dataset]
    suffix = ".jsonl"

    # Setup terashuf
    terashuf_dir = setup_terashuf(work_dir)

    # Download dataset
    if not skip_download:
        download_dataset(repo_id, src_dir, allow_patterns)
    else:
        orig_extension = ".jsonl"
        print(
            "Skipping download of dataset, make sure the dataset or jsonl files are present in the data directory"
        )
    if "fineweb" in dataset:
        parquet_to_jsonl(
            dataset, work_dir, src_dir, src_dir, preserve_subsets=preserve_subsets
        )

    # Set up environment variables
    os.environ["MEMORY"] = f"{memory}"
    os.environ["SEED"] = f"{seed}"

    # Run the original shuffling and splitting command
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")
    print(orig_extension, src_dir, cat_command, terashuf_executable)

    if preserve_subsets:
        for subset in os.listdir(f"{src_dir}"):
            # skip the non data directories
            if not os.path.isdir(f"{src_dir}/{subset}"):
                continue
            if subset in [".cache", ".git", ".github", "datatrove", "terashuf"]:
                continue
            prefix = f"{dataset}.{subset}.chunk."
            # Create validation set and remove lines from chunks
            validation_file = f"{out_dir}/{subset}/{dataset}.{subset}.val{suffix}"
            run_command(f"mkdir -p {out_dir}/{subset} ")
            run_command(
                f"ulimit -n 100000 "
                f"find {src_dir} -type f -name '*{subset}*{orig_extension}' -print0 | xargs -0 -I {{}} sh -c '{cat_command}' | {terashuf_executable} | "
                f" split -n l/{nchunks}  -d --suffix-length 2 --additional-suffix {suffix} - {out_dir}/{subset}/{prefix}"
                "; trap 'echo \"Caught signal 13, exiting with code 1\"; exit 1' PIPE;"
            )
            for i in range(nchunks):
                chunk_file = f"{out_dir}/{subset}/{prefix}{i:02d}{suffix}"
                run_command(f"head -n {k_validation} {chunk_file} >> {validation_file}")
                run_command(f"tail -n +{k_validation + 1} {chunk_file} > {chunk_file}.tmp && mv {chunk_file}.tmp {chunk_file}")
                if max_file_size:
                    # If file is larger than max_file_size, truncate it and remove the last line
                    run_command(f"""
                        if [ $(stat -c%s {chunk_file}) -gt {max_file_size} ]; then
                            truncate -s {max_file_size} {chunk_file}
                            tail -n 1 {chunk_file} | wc -c | xargs -I {{}} truncate {chunk_file} -s -{{}}
                        fi
                    """)
            if upload_to_hf:
                print("Uploading to Hugging Face...")
                run_command(
                    f"cd {out_dir}/{subset} && ls *.jsonl | parallel -j 4 --line-buffer 'echo {{}}; huggingface-cli upload {hf_path} {{}} {subset}/{{}} --repo-type dataset'"
                )
    else:
        run_command(
            f"ulimit -n 100000 && "
            f"find {src_dir} -type f -name '*{orig_extension}' -print0 | xargs -0 -I {{}} sh -c '{cat_command}' | {terashuf_executable} | "
            f" split -n l/{nchunks}  -d --suffix-length 2 --additional-suffix {suffix} - {out_dir}/{prefix}"
            "; trap 'echo \"Caught signal 13, exiting with code 1\"; exit 1' PIPE;"
        )

        # Create validation set and remove lines from chunks
        validation_file = f"{out_dir}/{dataset}.val{suffix}"
        for i in range(nchunks):
            chunk_file = f"{out_dir}/{prefix}{i:02d}{suffix}"
            run_command(f"head -n {k_validation} {chunk_file} >> {validation_file}")
            run_command(f"tail -n +{k_validation + 1} {chunk_file} > {chunk_file}.tmp && mv {chunk_file}.tmp {chunk_file}")
            if max_file_size:
                # If file is larger than max_file_size, truncate it and remove the last line
                run_command(f"""
                    if [ $(stat -c%s {chunk_file}) -gt {max_file_size} ]; then
                        truncate -s {max_file_size} {chunk_file}
                        tail -n 1 {chunk_file} | wc -c | xargs -I {{}} truncate {chunk_file} -s -{{}}
                    fi
                """)
        if upload_to_hf:
            print("Uploading to Hugging Face...")
            run_command(
                f"ls {out_dir}/*.jsonl | parallel -j 4 --line-buffer 'echo {{}}; huggingface-cli upload {hf_path} {{}} {{}} --repo-type dataset'"
            )

    print("All tasks completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("memory", type=float, default=8)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nchunks", type=int, default=32)
    parser.add_argument("--k_validation", type=int, default=10000)
    parser.add_argument(
        "--max_file_size",
        type=str,
        default=None,
        help="If specified, the dataset will be truncated to match this file size, e.g. 45G",
    )
    parser.add_argument(
        "--preserve_subsets",
        action="store_true",
        default=False,
        help="If true, the subsets will be preserved in the output directory",
    )
    parser.add_argument("--upload_to_hf", action="store_true", default=False)
    parser.add_argument("--hf_path", type=str, default=None)
    parser.add_argument("--skip_download", action="store_true", default=False)
    args = parser.parse_args()
    if args.upload_to_hf:
        if args.hf_path is None:
            raise ValueError("hf_path is required when upload_to_hf is true")

    main(
        args.dataset,
        args.memory,
        args.data_dir,
        args.seed,
        args.nchunks,
        args.preserve_subsets,
        args.upload_to_hf,
        args.hf_path,
        args.max_file_size,
        args.skip_download,
        args.k_validation,
    )
