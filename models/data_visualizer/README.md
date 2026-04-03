# Data Visualizer Demo

This demo ships one runnable agent specification that handles three cases:

1. The input data already matches the visualization script's expected schema.
2. The input data is not aligned, but can be converted by writing and running a new preprocessing script.
3. The input data cannot be converted safely, so the agent returns a structured explanation.

## Files

- `data_visualizer.agent`: the agent definition
- `requirements.txt`: dependencies installed into the demo virtual environment
- `scripts/visualize_data.py`: existing visualization script
- `scripts/write_preprocessor.py`: writes a new preprocessing script from a template
- `templates/monthly_csv_to_json.py.tmpl`: converter template used in the convertible branch
- `data/`: sample datasets for all three branches

## Provider Setup

`python -m agent.cli run` requires an OpenAI-compatible provider configuration.

Create `models/data_visualizer/.env` or a repository-level `.env`:

```bash
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your_api_key_here
```

## Run

```bash
python -m agent.cli run models/data_visualizer/data_visualizer.agent --verbose \
  "Visualize the aligned monthly sales data in models/data_visualizer/data/aligned_sales.json"
```

Optional example-focused prompt enrichment:

```bash
python -m agent.cli run models/data_visualizer/data_visualizer.agent --dspy --verbose \
  "Visualize the aligned monthly sales data in models/data_visualizer/data/aligned_sales.json"
```

The demo writes generated files to:

- `models/data_visualizer/.venv`
- `models/data_visualizer/generated`
- `models/data_visualizer/output`
