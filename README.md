# CSV Agent - AI-Powered CSV Analysis

A Python-based tool that enables AI-driven analysis of CSV data using local LLM inference through Ollama.

## Features

- Load and parse CSV files
- Get detailed dataset information
- Query data using natural language
- Generate visualizations
- Execute Python code for custom analysis
- Robust interactive mode with improved terminal handling
- MongoDB integration for data storage and natural language queries
- Integration with OpenWebUI (optional)

## Requirements

- Python 3.8+
- Ollama running locally (with models like llama3)
- MongoDB (for data storage capabilities)
- Optional: OpenWebUI for web interface

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/riccardomerolla/csv-agent.git
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

4. Optional: Start MongoDB if you plan to use the storage capabilities:
   ```bash
   mongod --dbpath /path/to/data/directory
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

### Interactive Mode

The interactive mode provides a robust REPL (Read-Eval-Print Loop) interface for analyzing your CSV files:

```bash
python cli.py --model deepseek-r1:8b --interactive
```

Interactive mode features:
- Proper handling of terminal input including special characters
- Robust exception handling to prevent session crashes
- Enhanced command history with readline support
- Input validation and cleaning

Example interactive session:
```
Starting interactive session with CSV Agent. Type 'exit' to end the session.

> /load data/sales.csv
CSV file 'data/sales.csv' loaded successfully as 'sales'

> /info
Dataset: sales
Rows: 5000, Columns: 8

Columns:
- date (object): 365 unique values, 0 missing
- product_id (int64): 150 unique values, 0 missing
- category (object): 5 unique values, 0 missing
- price (float64): 120 unique values, 0 missing
- quantity (int64): 15 unique values, 0 missing

> What's the average sales by category?

[Agent responds with analysis and executes code...]

> /exit
Ending interactive session.
```

Available commands:
- `load <file_path> [dataset_name]` - Load a CSV file
- `info [dataset_name]` - Show dataset information
- `ask <question>` - Ask a natural language question about the data
- `viz <type> <column1> [column2...]` - Create a visualization
- `exec <code>` - Execute Python code on the current dataframe
- `list` - List all loaded datasets
- `switch <dataset_name>` - Switch to a different dataset
- `store <collection_name>` - Store DataFrame in MongoDB collection
- `filter <collection_name> <query>` - Query MongoDB using natural language
- `clear` - Clear the conversation history
- `help` - Display available commands
- `exit` or `quit` - Exit the application

Examples:
```
(csv_agent) load customer_data.csv
(csv_agent) info
(csv_agent) ask What is the average age of customers by region?
(csv_agent) viz histogram age
(csv_agent) viz scatter age income
(csv_agent) store customers
(csv_agent) filter customers find all customers older than 30 in California
```

### MongoDB Integration and Natural Language Queries

#### Storing Data in MongoDB

Use the `store` command to persist your data in MongoDB for later retrieval:

```
(csv_agent) store customers
```

This will store the current dataframe in a MongoDB collection named "customers".

#### Using the Filter Command

The `filter` command allows you to query MongoDB collections using natural language instead of learning complex MongoDB query syntax.

Syntax:
```
filter <collection_name> <natural language query>
```

##### How It Works

When using the filter command:
1. The natural language query is sent to the LLM
2. The LLM translates your request into a proper MongoDB query
3. The query is executed against your MongoDB collection
4. Results are returned as a pandas DataFrame

##### Examples

Basic filtering:
```
(csv_agent) filter customers find all customers from New York
```

Compound conditions:
```
(csv_agent) filter customers find customers who spent more than $1000 and joined before 2022
```

Sorting and limiting:
```
(csv_agent) filter customers find the top 5 customers by purchase amount
```

Aggregate queries:
```
(csv_agent) filter sales give me monthly sales totals for each product category
```

Field selection:
```
(csv_agent) filter customers return only the name and email of customers who haven't made a purchase in 6 months
```

Advanced filtering:
```
(csv_agent) filter transactions find all transactions with unusual spending patterns where amount exceeds twice the customer's average
```

Comparison filters:
```
(csv_agent) filter products find all products with inventory below 10 units that have high demand (sold > 50 units last month)
```

#### Additional Filter Features

- Results are automatically converted to pandas DataFrames for further analysis
- The generated MongoDB queries are displayed to help you learn MongoDB syntax
- You can modify the returned data with standard DataFrame operations
- Error handling provides clear feedback if your query can't be processed

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
- MongoDB for data storage and retrieval
- A conversational interface for data analysis

When you ask a question, the agent:
1. Analyzes the dataset structure
2. Formulates a context-rich prompt for the LLM
3. Processes the LLM's response
4. Executes any code if necessary
5. Returns results and/or visualizations

For MongoDB queries, the agent:
1. Takes your natural language query
2. Converts it to a MongoDB query using the LLM
3. Executes the query against your MongoDB collection
4. Returns the matching documents

## Extending the Agent

You can extend this agent by:
- Adding more visualization types
- Implementing caching for LLM responses
- Adding support for data preprocessing and transformation
- Implementing export capabilities for analysis results
- Adding support for SQL queries against the data

## License

This project is licensed under the MIT License - see the LICENSE file for details.
`