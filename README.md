# CSV Agent - AI-Powered CSV Analysis

A Python-based tool that enables AI-driven analysis of CSV data using local LLM inference through Ollama.

## Features

- Load and parse CSV files
- Get detailed dataset information
- Query data using natural language
- Generate visualizations
- Execute Python code for custom analysis
- Integration with OpenWebUI (optional)

## Requirements

- Python 3.8+
- Ollama running locally (with models like llama3)
- Optional: OpenWebUI for web interface

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/csv-agent.git
   cd csv-agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure Ollama is running with your preferred model:
   ```bash
   ollama run llama3
   ```

## Usage

### Command Line Interface

Start the command line interface:

```bash
python cli.py --model llama3
```

```bash
python cli.py --model deepseek-r1:8b
```

Available commands:
- `load <file_path> [dataset_name]` - Load a CSV file
- `info [dataset_name]` - Show dataset information
- `ask <question>` - Ask a natural language question about the data
- `viz <type> <column1> [column2...]` - Create a visualization
- `exec <code>` - Execute Python code on the current dataframe
- `list` - List all loaded datasets
- `switch <dataset_name>` - Switch to a different dataset
- `exit` or `quit` - Exit the application

Examples:
```
(csv_agent) load customer_data.csv
(csv_agent) info
(csv_agent) ask What is the average age of customers by region?
(csv_agent) viz histogram age
(csv_agent) viz scatter age income
```

### OpenWebUI Integration

Start the OpenWebUI integration server:

```bash
python openwebui_integration.py --port 5000 --openwebui-url http://localhost:8080
```

This creates an API endpoint that can be integrated with OpenWebUI for a web-based interface to the CSV agent.

## How It Works

The CSV Agent combines:
- Pandas for data processing
- Ollama for LLM inference
- Matplotlib and Seaborn for visualizations
- A conversational interface for data analysis

When you ask a question, the agent:
1. Analyzes the dataset structure
2. Formulates a context-rich prompt for the LLM
3. Processes the LLM's response
4. Executes any code if necessary
5. Returns results and/or visualizations

## Extending the Agent

You can extend this agent by:
- Adding more visualization types
- Implementing caching for LLM responses
- Adding support for data preprocessing and transformation
- Implementing export capabilities for analysis results
- Adding support for SQL queries against the data

## License

This project is licensed under the MIT License - see the LICENSE file for details.