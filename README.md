# K2-Think-Inference

## Dependencies

You will need to install the following python packages:
```bash
pip install -r requirements.txt
```

## Configuration
You need to set the following arguments in `main.py` under `class Env`.
```python
# Endpoint configuration of planner llm
PLANNER_LLM_API_KEY: str = ''
PLANNER_LLM_BASE_URL: str = ''
PLANNER_LLM_MODEL: str = ''

# Endpoint configuration of solver llm
SOLVER_LLM_API_KEY: str = ''
SOLVER_LLM_BASE_URL: str = ''
SOLVER_LLM_MODEL: str = "K2-Think"
```
The planner llm will be responsible for extracting topics, generating plans, and comparing answer pairs. You may choose any OpenAI compatible endpoint you want to act as the planner. For example, you can start a [vllm](https://docs.vllm.ai/en/stable/) localhost endpoint of any huggingface model. The following script will start an endpoint serving [Qwen/Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B):
```bash
vllm serve Qwen/Qwen3-235B-A22B \
 --tensor_parallel_size 8 \
 --served-model-name Qwen/Qwen3-235B-A22B \
 --port 8080
```
After the endpoint is up and running on localhost, you can set the following arguments:
```python
# Endpoint configuration of planner llm
PLANNER_LLM_API_KEY: str = ''
PLANNER_LLM_BASE_URL: str = 'http://localhost:8080/v1'
PLANNER_LLM_MODEL: str = 'Qwen/Qwen3-235B-A22B'
```
Similarly, you can choose your favorite reasoning model for the solver llm, which is responsible for solving the input problems.

## Test the Script
```bash
python main.py
```
You can change `query` in line `257` from `main.py` to test any problems you like.
