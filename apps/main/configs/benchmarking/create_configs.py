from pathlib import Path

base_dir = Path(__file__).parent
# config_string = """dump_dir: /root/lingua_dump/train/benchmarking/llama_{s}B_bs{bs}_gas{gas}/
config_string = """dump_dir: /scratch/gsa/lingua_dump/train/benchmarking/llama_{s}B_bs{bs}_gas{gas}/
name: "llama_{s}B_bs{bs}_gas{gas}"

grad_acc_steps: {gas}
data:
  batch_size: {bs}"""

for bs in [1, 2, 4, 8]:
    for gas in [1, 2, 4, 8]:
        for s in [1, 7]:
            with open(base_dir / f"llama_{s}B_bs{bs}_gas{gas}.yaml", "w") as f:
                f.write(config_string.format(s=s, bs=bs, gas=gas))

# python apps/main/configs/benchmarking/create_configs.py
# python -m lingua.amdtool script=apps.main.train config=["apps/main/configs/benchmarking/llama_7B_base.yaml","apps/main/configs/benchmarking/llama_7B_bs1_gas1.yaml"] copy_code=False