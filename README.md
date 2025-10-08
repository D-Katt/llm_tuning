# LLM fine-tuning

Example of LLM fine-tuning (domain adaptation).

Install dependencies for Axolotl (see documentation here: https://github.com/axolotl-ai-cloud/axolotl).

Save train and validation datasets to "data" directory. Datasets should be in a standard chat format:

```
[
  {
    "messages": [
      {
        "role": "user",
        "content": "User question"
      },
      {
        "role": "assistant",
        "content": "Expected answer"
      }
    ]
  },
]
```

For LoRA fine-tuning:
```commandline
python3 -m axolotl.cli.train src/lora_config.yaml
```

Full SFT fine-tuning:
```commandline
python3 -m axolotl.cli.train src/full_config.yaml
```
