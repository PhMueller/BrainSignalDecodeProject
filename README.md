## How to install

### Via pip:

```shell
cd <ProjectDir>
pip install .
```

### Run Baseline:
1. Adapt the `data_dir` and the `output_dir` in the setting file in \
   `./brain-brain_decode_project/library/AgeBaseline.yaml`


2. Run the configuration library/AgeBaseline.yaml via
    ```shell
    python brain_decode_project/runner/run_from_config.py \
      --yaml ./brain-brain_decode_project/library/AgeBaseline.yaml \
      --run_config RUN_Baseline
    ```