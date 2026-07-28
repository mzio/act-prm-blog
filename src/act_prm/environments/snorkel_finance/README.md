# Snorkel Finance Environment

Tool-calling environment for financial question answering over real company filings.
Uses data from the [FinQABenchmark](https://github.com/snorkel-ai/FinQABenchmark).

## Data Setup

### 1. Clone the FinQABenchmark repo

```bash
mkdir -p data/snorkel_finance && cd data/snorkel_finance
git clone https://github.com/snorkel-ai/FinQABenchmark _repo
```

### 2. Copy the raw table data

The backend expects a flat directory of company folders, each containing JSON table files,
plus a top-level `tables_cleaned_all_companies.json` metadata file.

```bash
cp -r _repo/data/raw ./raw
```

Expected structure:

```
data/snorkel_finance/
├── raw/                                  ← data_path in configs
│   ├── tables_cleaned_all_companies.json
│   ├── alphabet/
│   │   ├── <table1>.json
│   │   └── ...
│   ├── amazon/
│   ├── apple/
│   ├── at_t/
│   ├── berkshire/
│   ├── boa/
│   ├── boeing/
│   └── ... (20 companies total)
└── _repo/                                ← cloned FinQABenchmark repo
```

### 3. Get the benchmark CSVs

The benchmark CSVs contain the questions and ground-truth answers. They ship with the
FinQABenchmark repo:

```bash
mkdir -p benchmark
cp _repo/data/benchmark/finqa.csv benchmark/
cp _repo/data/benchmark/finqa_reasoning.csv benchmark/
```

Or all together now:

```bash
mkdir -p data/snorkel_finance && cd data/snorkel_finance && git clone
https://github.com/snorkel-ai/FinQABenchmark _repo && cp -r _repo/data/raw ./raw && mkdir -p benchmark && cp
_repo/data/benchmark/finqa.csv _repo/data/benchmark/finqa_reasoning.csv benchmark/
```

Each CSV has columns: `id`, `company`, `question` (or `user_query`), `answer`.

Expected structure now:

```
data/snorkel_finance/
├── raw/                                  ← data_path in configs
│   ├── tables_cleaned_all_companies.json
│   ├── alphabet/
│   │   ├── <table1>.json
│   │   └── ...
│   ├── amazon/
│   ├── apple/
│   ├── at_t/
│   ├── berkshire/
│   ├── boa/
│   ├── boeing/
│   └── ... (20 companies total)
├── benchmark/                            ← benchmark CSVs
│   ├── finqa.csv                         (290 quantitative questions)
│   └── finqa_reasoning.csv               (79 qualitative reasoning questions)
└── _repo/                                ← cloned FinQABenchmark repo
```

## Tasks

| Task | Questions | Answer format | Example |
|------|-----------|---------------|---------|
| `finqa` | 290 | Boxed numeric: `\boxed{6.118}` | "What was the revenue growth rate?" |
| `finqa_reasoning` | 79 | Free-text paragraph | "Explain AT&T's postretirement cost trends" |

## Architecture

```
env.py       ← SnorkelFinanceEnv: reset/step loop, split management, grading
backend.py   ← DataBackend: loads JSON tables into SQLite, executes filtered queries
tools.py     ← 5 tools: get_descriptions, get_table_info, sql_query, calculator, respond_user
prompts.py   ← Task-specific system prompts
```

### Tools

| Tool | Description |
|------|-------------|
| `get_descriptions` | List available tables for a company |
| `get_table_info` | Get column names, dtypes, and unique values for a table |
| `sql_query` | Execute a SELECT query on a company's table (with safety filters) |
| `calculator` | Evaluate Python math expressions (uses `math` module) |
| `respond_user` | Submit final answer — triggers grading and ends the episode |

### SQL Safety

The `sql_query` tool enforces:
- SELECT-only (no INSERT/UPDATE/DELETE/DROP)
- Must include a WHERE, HAVING, IN, LIKE, or BETWEEN clause (no `SELECT *` dumps)
- Queries run in an ephemeral in-memory SQLite database

### Grading

- **finqa**: 2-decimal-place truncation comparison (e.g., answer=0.112, response=0.1129 → correct since both truncate to 0.11)
- **finqa_reasoning**: LLM-based grading via `SnorkelFinanceGrader` (requires `ANTHROPIC_API_KEY`)

## Config Files

**Online RL** (interactive environment with tool calls):
```
configs/environments/act_prm/snorkel_finance.yaml
configs/environments/act_prm/snorkel_finance_fs1.yaml        # 1 few-shot example
configs/environments/act_prm/snorkel_finance_aligned.yaml     # aligned train/eval/test splits
```

**SFT / Language Modeling** (static traces from HuggingFace):
```
configs/environments/act_lm/snorkel_finance_gt.yaml           # ground truth traces
configs/environments/act_lm/snorkel_finance_aprm*.yaml        # APRM-generated traces
```

## Usage

### In training scripts

```bash
# RL with snorkel finance
uv run python main_pytorch.py \
--env_config act_prm/snorkel_finance \
--generator_config default \
--trainer_config <your_trainer> \
...

# SFT on ground truth traces
uv run python main_pytorch.py \
--env_config act_lm/snorkel_finance_gt \
--trainer_config sft \
...
```

### Standalone

```python
from strl.environments.snorkel_finance import SnorkelFinanceEnv

env = SnorkelFinanceEnv(
    data_path="data/snorkel_finance/raw",
    benchmark_csv="data/snorkel_finance/benchmark/finqa_reasoning.csv",
    task="finqa_reasoning",
    max_turns=50,
    split="train",
)
env.init_data()

state = env.reset(sample_idx=0)
print(state.messages)   # system prompt + user question
print(state.tools)      # 5 tool schemas
```

## Tests

```bash
# Unit tests (no API key needed, but requires data files)
pytest tests/test_snorkel_finance.py -v -k "not live"

# Live grader tests (requires ANTHROPIC_API_KEY)
pytest tests/test_snorkel_finance.py -v -m live
```

Tests will skip gracefully if `data_path` or benchmark CSVs are not found.
