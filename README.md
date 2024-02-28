<img src="assets/ldb-logo5.webp" alt="LDB" width="115" align="left"><div align="center"><h1>LDB: A Large Language Model Debugger via Verifying Runtime Execution Step by Step</h1></div>

<p align="center">
  <a href="assets/LDB_paper.pdf">
    <img src="https://img.shields.io/badge/📝-Paper-blue">
  </a>
  <a href="https://github.com/FloridSleeves/LLMDebugger">
    <img src="https://img.shields.io/badge/👩‍💻-Code-green">
  </a>
</p>

This repository contains the code and dataset for our paper **LDB: A Large Language Model Debugger via Verifying Runtime Execution Step by Step**.

We introduce 🛠️**LDB**, a novel debugging framework that enables LLMs to refine their generated programs with the runtime execution information. Specifically, LDB immitates how human developers debug programs. It segments the programs into basic blocks and tracks the values of intermediate variables after each block throughout the runtime execution. This allows LLMs to concentrate on simpler code units within the overall execution flow, verify their correctness against the task description block by block, and efficiently pinpoint any potential errors.

![image](assets/overview-ldb.png)

## 📦 Installation

```bash
conda create -n ldb python=3.10
conda activate ldb
python -m pip install -r requirements.txt
```

## 📈 Usage

### Set Environment

If you use OpenAI models as backbones:

```bash
export OPENAI_API_KEY=[your OpenAI API Key]
```

If you use `starcoder` or `codellama`, we recommend to setup an OpenAI compatible server based on vLLM. Here is the instruction [Setup vLLM backbones](#setup-vllm-backbones).

### Generate Program Seeds

```bash
cd ./programming
./run_simple.sh [dataset] [model] [output_dir]
```

The result is in `output_data/simple/[dataset]/[model]/[output_dir]`.

Available options:

| Option  | Value                                                                        |
| ------- | ---------------------------------------------------------------------------- |
| dataset | `humaneval`, `mbpp`, `transcoder`                                                                            |
| model   | `gpt-3.5-turbo-0613`, `gpt-4-1106-preview`, `starcoder`, `codellama` (codellama/CodeLlama-34b-Instruct-hf)   |

### Debug Programs

Run the script:

```bash
cd ./programming
./run_ldb.sh [dataset] [model] [seed] [output_dir]
```

The result is in `output_data/ldb/[dataset]/[model]/[output_dir]`

Available options:

| Option  | Value|
| ------- | --------------------------------------------------------------------|
| dataset | `humaneval`, `mbpp`, `transcoder`  |
| model   | `gpt-3.5-turbo-0613`, `gpt-4-1106-preview`, `starcoder`, `codellama` (codellama/CodeLlama-34b-Instruct-hf)|
| seed    | Path to the seed program you want to debug. You can find the seed programs we use in experiments in `input_data/[dataset]/seed/[model]/seed.jsonl`.|

### Setup vLLM backbones

We use the OpenAI compatible server based on vLLM. Please refer [OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#openai-compatible-server) for detailed instructions to setup the local servers. To start the server:
```bash
python -m vllm.entrypoints.openai.api_server --model bigcode/starcoder
```
LDB automatically sets up the connection to your local servers when you specify model `starcoder` or `codellama`.

If your server port is not the default `8000`, please set the option `--port` in `run_simple.sh` or `run_ldb.sh` to your local server port.

## 🐞 Bugs or Questions?

If you have any questions, feel free to post issues in this repo.

## 📑 Citation

If you find our work helpful, please cite us:
```
@misc{zhong2024ldb,
      title={LDB: A Large Language Model Debugger via Verifying Runtime Execution Step-by-step}, 
      author={Li Zhong and Zilong Wang and Jingbo Shang},
      year={2024},
      eprint={2402.16906},
      archivePrefix={arXiv},
      primaryClass={cs.SE}
}
```
