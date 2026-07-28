# Snorkel Insurance Underwriting Environment

Tool-calling environment for commercial small-business insurance underwriting.
Uses data from the [multi-turn-insurance-underwriting-benchmark-generation](https://github.com/snorkel-ai/multi-turn-insurance-underwriting-benchmark-generation).

## Data Setup

### 1. Clone the benchmark repo

```bash
mkdir -p data/snorkel_insurance && cd data/snorkel_insurance
git clone https://github.com/snorkel-ai/multi-turn-insurance-underwriting-benchmark-generation _repo
```

### 2. Copy the resource data (parquet files)

The backend builds an in-memory SQLite database from these parquet files.

```bash
cp -r _repo/resources ./resources
```

### 3. Copy the tool data

```bash
cp -r _repo/tool_data ./tool_data
```

### 4. Copy the task data

```bash
mkdir -p task_data
cp _repo/task_data/downsampled_task_set.json task_data/
```

Or all together now:

```bash
mkdir -p data/snorkel_insurance && cd data/snorkel_insurance && git clone 
https://github.com/snorkel-ai/multi-turn-insurance-underwriting-benchmark-generation _repo && cp -r
_repo/resources ./resources && cp -r _repo/tool_data ./tool_data && mkdir -p task_data && cp
_repo/task_data/downsampled_task_set.json task_data
```

Expected structure:

```
data/snorkel_insurance/
├── resources/                              ← parquet files → SQLite
│   ├── naics_table_2022.parquet
│   ├── naics_2digit_table_2022.parquet
│   ├── naics_2022_2017_conversion.parquet
│   ├── naics_2012_2017_conversion.parquet
│   ├── sba_size_standards_by_2012_naics.parquet
│   ├── small_business_insurance_appetite.parquet
│   ├── small_business_lobs.parquet
│   └── states.parquet
├── tool_data/                              ← guidelines + metadata
│   ├── underwriting_rules.txt
│   ├── table_dictionary.json
│   ├── table_data_dictionaries.json
│   └── building_construction_types.json
├── task_data/                              ← benchmark tasks
│   └── downsampled_task_set.json           (300 tasks)
└── _repo/                                  ← cloned benchmark repo
```

## Tasks

| Task Type | Count | Description |
|-----------|-------|-------------|
| `appetite_determination` | 81 | Is this LOB in appetite for the company? |
| `lob_recommendation` | 77 | What LOBs should we offer? |
| `limits_recommendation` | 60 | What policy limits to recommend? |
| `small_business_eligibility` | 47 | Does this company qualify as a small business? |
| `deductible_recommendation` | 20 | What deductible to recommend? |
| `naics_classification` | 15 | What is this company's NAICS code? |

## Architecture

```
env.py       ← SnorkelInsuranceEnv: reset/step loop, split management, grading
backend.py   ← DataBackend: loads parquet into SQLite, serves guidelines
tools.py     ← 7 tools: get_underwriting_guidelines, get_table_descriptions,
               get_table_data_dictionary, list_tables, get_table_schema,
               read_query, respond_user
prompts.py   ← System prompt and user prompt template
```

### Tools

| Tool | Description |
|------|-------------|
| `get_underwriting_guidelines` | Get the full underwriting rules document |
| `get_table_descriptions` | Get descriptions of all database tables |
| `get_table_data_dictionary` | Get column-level metadata for a table |
| `list_tables` | List all tables in the database |
| `get_table_schema` | Get PRAGMA table_info for a table |
| `read_query` | Execute a read-only SQL SELECT query (max 500 rows) |
| `respond_user` | Submit final answer — triggers grading and ends episode |

### SQL Safety

The `read_query` tool enforces:
- SELECT-only (no INSERT/UPDATE/DELETE/DROP/ALTER/CREATE)
- No multi-statement queries
- Results capped at 500 rows

### Grading

LLM-based grading via `SnorkelInsuranceGrader` (requires `ANTHROPIC_API_KEY`).
Evaluates whether the agent's answer matches the programmatic reference answer
derived from each task's ground-truth fields.

## Usage

### Standalone

```python
from dplm.environments.snorkel_insurance import SnorkelInsuranceEnv

env = SnorkelInsuranceEnv(
    data_path="data/snorkel_insurance",
    task_data_path="data/snorkel_insurance/task_data/downsampled_task_set.json",
    max_turns=50,
    split="train",
)

state = env.reset(sample_idx=0)
print(state.new_messages)   # system prompt + user question with company info
print(state.tools)          # 7 tool schemas
```
