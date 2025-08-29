# K2-Think-Inference

## Dependencies

You will need to install the following python packages:
```bash
pip install openai pydantic
```

## Configuration
You need to set the following arguments in `main.py`.
```python
SOLVER_LLM_API_KEY: str = ''
SOLVER_LLM_BASE_URL: str = ''
SOLVER_LLM_MODEL: str = "K2-Think"

PLANNER_LLM_API_KEY: str = ''
PLANNER_LLM_BASE_URL: str = ''
PLANNER_LLM_MODEL: str = ''
```

## Test the Script
```bash
python main.py
```
