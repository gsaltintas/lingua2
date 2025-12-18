config_string = """dump_dir: /fsx/craffel/lingua_logs/benchmarking/llama_{s}B_bs{bs}_gas{gas}/
name: "llama_{s}B_bs{bs}_gas{gas}"

grad_acc_steps: {gas}
data:
  batch_size: {bs}"""

for bs in [1, 2, 4, 8]:
    for gas in [1, 2, 4, 8]:
        for s in [1, 7]:
            with open(f"llama_{s}B_bs{bs}_gas{gas}.yaml", "w") as f:
                f.write(config_string.format(s=s, bs=bs, gas=gas))
