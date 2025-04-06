import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import re
from typing import Dict, List, Optional, Union, Any, Tuple
import requests
from python_repl import PythonREPL
import readline  # Add readline for better terminal input handling

class CSVAgent:
    def __init__(self, model_name="llama3", ollama_host="http://localhost:11434", max_tokens=2000):
        """
        Initialize the CSV agent.
        
        Args:
            model_name: The Ollama model to use
            ollama_host: URL for the Ollama API
            max_tokens: Maximum tokens for model response
        """
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.max_tokens = max_tokens
        self.dataframes = {}
        self.current_df_name = None
        self.chat_history = []
        self.repl = PythonREPL()  # Initialize REPL immediately
        self.session_active = False
        
    def load_csv(self, file_path: str, name: Optional[str] = None) -> bool:
        """
        Load a CSV file into a pandas DataFrame.
        
        Args:
            file_path: Path to the CSV file
            name: Optional name for the DataFrame
            
        Returns:
            bool: True if loading was successful
        """
        try:
            if name is None:
                name = os.path.basename(file_path).split('.')[0]
                
            # Try to automatically detect encoding and delimiter
            df = pd.read_csv(file_path)
            
            # Store the dataframe
            self.dataframes[name] = df
            self.current_df_name = name
            
            # Initialize or update REPL with the current dataframe
            self.repl.set_variable('df', df)
            
            print(f"CSV file '{file_path}' loaded successfully as '{name}'")
            return True
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return False
    
    def get_dataframe_info(self, df_name: Optional[str] = None) -> Dict:
        """
        Get information about the DataFrame.
        
        Args:
            df_name: Name of the DataFrame to analyze
            
        Returns:
            Dict with DataFrame metadata
        """
        df_name = df_name or self.current_df_name
        if df_name not in self.dataframes:
            return {"error": f"DataFrame '{df_name}' not found"}
        
        df = self.dataframes[df_name]
        
        # Collect basic information
        info = {
            "name": df_name,
            "shape": {
                "rows": df.shape[0],
                "columns": df.shape[1]
            },
            "columns": [],
            "sample_data": df.head(5).to_dict(orient='records')
        }
        
        # Analyze each column
        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "missing_values": int(df[col].isna().sum()),
                "unique_values": int(df[col].nunique())
            }
            
            # Add statistics based on data type
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update({
                    "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    "median": float(df[col].median()) if not pd.isna(df[col].median()) else None
                })
            elif pd.api.types.is_string_dtype(df[col]):
                # Get top 5 most common values
                value_counts = df[col].value_counts().head(5).to_dict()
                col_info["top_values"] = {str(k): int(v) for k, v in value_counts.items()}
            
            info["columns"].append(col_info)
        
        return info
    
    def query_data(self, query: str, df_name: Optional[str] = None, include_history: bool = True) -> str:
        """
        Process a natural language query about the data using Ollama.
        
        Args:
            query: Natural language query from the user
            df_name: Name of the DataFrame to query
            include_history: Whether to include chat history in the prompt
            
        Returns:
            Response from the model
        """
        df_name = df_name or self.current_df_name
        if df_name not in self.dataframes:
            return f"Error: DataFrame '{df_name}' not found"
        
        # Update REPL with the current dataframe
        self.repl.set_variable('df', self.dataframes[df_name])
        
        df_info = self.get_dataframe_info(df_name)
        
        # Construct the history part of the prompt
        history_text = ""
        if include_history and len(self.chat_history) > 0:
            history_text = "Previous conversation:\n"
            for entry in self.chat_history[-5:]:  # Include last 5 interactions to avoid token limits
                role = "User" if entry["role"] == "user" else "Assistant"
                history_text += f"{role}: {entry['content']}\n"
                if "code" in entry and entry["code"]:
                    history_text += f"Code executed:\n```python\n{entry['code']}\n```\n"
                if "result" in entry and entry["result"]:
                    result_text = entry["result"].get("output", "")
                    if result_text:
                        history_text += f"Result: {result_text}\n"
            history_text += "\n"
        
        # Construct the main prompt
        prompt = f"""
You are an agent that writes and executes python code
You have access to a Python, which you can use to execute the python code.
You must write the python code assuming that the dataframe (stored as df) has already been read.
You must write the python code that print each steps like a notebook.
You must write the code in a way that it can be executed in a Python REPL.
You must write the code assuming that the df variable is a pandas dataframe and is already defined.
If you get an error, debug your code and try again.
You might know the answer without running any code, but you should still run the code to get the answer.
If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
Do not create example dataframes

{history_text}

TOOLS:

------

You have access to the following tools:

pandas, matplotlib, seaborn

To use a tool, please use the following format:

```

Thought: Do I need to use a tool? Yes

Action: the action to take, should be one of [pandas, matplotlib, seaborn]

Action Input: the input to the action

Observation: the result of the action

```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```

Thought: Do I need to use a tool? No

Final Answer: [your response here]

```

Begin!

Here is information about the dataset:
- Name: {df_info['name']}
- Shape: {df_info['shape']['rows']} rows and {df_info['shape']['columns']} columns
- Columns: {', '.join([c['name'] for c in df_info['columns']])}

Answer the following query about the dataset:

{query}
"""

        print(prompt)
        # Add this query and response to chat history
        self.chat_history.append({"role": "user", "content": query})
        
        # Call Ollama API
        response = self._call_ollama(prompt)
        
        # Add to chat history
        self.chat_history.append({"role": "assistant", "content": response})
        
        return response
    
    def execute_code(self, code: str, df_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute Python code on the DataFrame using PythonREPL.
        
        Args:
            code: Python code to execute
            df_name: Name of the DataFrame to use
            
        Returns:
            Dict with execution results and possible plot path
        """
        df_name = df_name or self.current_df_name
        if df_name not in self.dataframes:
            return {"error": f"DataFrame '{df_name}' not found"}
        
        # Update the dataframe in the REPL
        self.repl.set_variable('df', self.dataframes[df_name])
        
        # Execute the code using the REPL
        result = self.repl.execute(code)
        
        # Add execution result to chat history
        self.chat_history.append({"role": "assistant", "code": code, "result": result})
        
        return result
    
    def visualize(self, viz_type: str, columns: List[str], df_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a visualization based on specified parameters using PythonREPL.
        
        Args:
            viz_type: Type of visualization (histogram, scatter, etc.)
            columns: Columns to include in the visualization
            df_name: Name of the DataFrame to visualize
            
        Returns:
            Dict with plot path or error
        """
        df_name = df_name or self.current_df_name
        if df_name not in self.dataframes:
            return {"error": f"DataFrame '{df_name}' not found"}
        
        df = self.dataframes[df_name]
        
        # Validate columns
        for col in columns:
            if col not in df.columns:
                return {"error": f"Column '{col}' not found in DataFrame"}
        
        # Generate a unique plot name
        plot_id = f"{df_name}_{viz_type}_{hash('_'.join(columns)) % 10000}"
        plot_path = f"viz_{plot_id}.png"
        
        # Update the dataframe in the REPL
        self.repl.set_variable('df', df)
        
        # Create visualization code based on type
        code = "import matplotlib.pyplot as plt\nimport seaborn as sns\n"
        
        if viz_type == "histogram":
            if len(columns) == 1:
                code += f"plt.figure(figsize=(10, 6))\n"
                code += f"plt.hist(df['{columns[0]}'].dropna())\n"
                code += f"plt.xlabel('{columns[0]}')\n"
                code += f"plt.ylabel('Frequency')\n"
                code += f"plt.title('Histogram of {columns[0]}')\n"
                code += f"plt.tight_layout()\n"
                code += f"plt.savefig('{plot_path}')\n"
                code += "plt.close()"
            else:
                return {"error": "Histogram requires exactly one column"}
                
        elif viz_type == "scatter":
            if len(columns) == 2:
                code += f"plt.figure(figsize=(10, 6))\n"
                code += f"plt.scatter(df['{columns[0]}'], df['{columns[1]}'])\n"  # Fixed syntax error with extra quote
                code += f"plt.xlabel('{columns[0]}')\n"
                code += f"plt.ylabel('{columns[1]}')\n"
                code += f"plt.title('Scatter plot of {columns[0]} vs {columns[1]}')\n"
                code += f"plt.tight_layout()\n"
                code += f"plt.savefig('{plot_path}')\n"
                code += "plt.close()"
            else:
                return {"error": "Scatter plot requires exactly two columns"}
                
        elif viz_type == "bar":
            code += f"plt.figure(figsize=(10, 6))\n"
            if len(columns) == 1:
                code += f"df['{columns[0]}'].value_counts().head(10).plot(kind='bar')\n"
                code += f"plt.title('Bar chart of {columns[0]} (top 10)')\n"
            elif len(columns) == 2:
                code += f"df.groupby('{columns[0]}')['{columns[1]}'].mean().sort_values(ascending=False).head(10).plot(kind='bar')\n"
                code += f"plt.title('Mean {columns[1]} by {columns[0]} (top 10)')\n"
            else:
                return {"error": "Bar chart requires one or two columns"}
            code += f"plt.tight_layout()\n"
            code += f"plt.savefig('{plot_path}')\n"
            code += "plt.close()"
        
        elif viz_type == "line":
            code += f"plt.figure(figsize=(10, 6))\n"
            if len(columns) >= 1:
                if len(columns) == 1:
                    code += f"df['{columns[0]}'].plot(kind='line')\n"
                    code += f"plt.title('Line plot of {columns[0]}')\n"
                else:
                    x_col = columns[0]
                    for y_col in columns[1:]:
                        code += f"df.sort_values('{x_col}').plot(x='{x_col}', y='{y_col}', figsize=(10, 6))\n"
                    code += f"plt.title('Line plot of {', '.join(columns[1:])} vs {x_col}')\n"  # Fix this line
            else:
                return {"error": "Line plot requires at least one column"}
            code += f"plt.tight_layout()\n"
            code += f"plt.savefig('{plot_path}')\n"
            code += "plt.close()"
        
        elif viz_type == "boxplot":
            if len(columns) >= 1:
                code += f"plt.figure(figsize=(10, 6))\n"
                code += f"sns.boxplot(data=df[{columns}])\n"
                code += f"plt.title('Box plot of {', '.join(columns)}')\n"
                code += f"plt.tight_layout()\n"
                code += f"plt.savefig('{plot_path}')\n"
                code += "plt.close()"
            else:
                return {"error": "Box plot requires at least one column"}
        
        else:
            return {"error": f"Unsupported visualization type: {viz_type}"}
        
        # Execute the visualization code using the REPL
        result = self.repl.execute(code)
        
        if result["error"]:
            return {"error": result["error"]}
        elif os.path.exists(plot_path):
            return {"plot_path": plot_path}
        else:
            return {"error": "Failed to generate plot"}
    
    def _call_ollama(self, prompt: str) -> str:
        """
        Call the Ollama API.
        
        Args:
            prompt: The prompt to send to Ollama
            
        Returns:
            Model response as text
        """
        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "max_tokens": self.max_tokens
            }
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            return f"Error calling Ollama API: {e}"
    
    def _extract_code_blocks(self, text: str) -> List[str]:
        """
        Extract Python code blocks from a text.
        
        Args:
            text: Text that may contain code blocks
            
        Returns:
            List of extracted code blocks
        """
        # Pattern for code blocks with triple backticks
        blocks = []
        
        # Find blocks marked with ```python or just ```
        pattern = r"```(?:python)?\s*(.*?)\s*```"
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            blocks.extend(matches)
        
        return blocks
        
    def start_interactive_session(self):
        """
        Start an interactive REPL session with the agent.
        """
        self.session_active = True
        print("Starting interactive session with CSV Agent. Type 'exit' to end the session.")
        
        while self.session_active:
            try:
                # Use readline for better terminal input handling
                user_input = input("\n> ")
                
                # Clean the input by removing carriage returns and other control characters
                user_input = re.sub(r'[\r\n]+', '', user_input).strip()
                
                if user_input.lower() in ['exit', 'quit']:
                    self.session_active = False
                    print("Ending interactive session.")
                    break
                    
                # Process commands
                if user_input.startswith("/"):
                    self._process_command(user_input[1:])
                    continue
                
                if not user_input:  # Skip empty inputs
                    continue
                    
                # Process query
                response = self.query_data(user_input)
                print("\n" + response)
                
                # Check if there's code to execute
                code_blocks = self._extract_code_blocks(response)
                if code_blocks:
                    print("\nExecuting code...")
                    for code in code_blocks:
                        result = self.execute_code(code)
                        
                        if result.get("error"):
                            print(f"Execution error: {result['error']}")
                        elif result.get("output"):
                            print("\nOutput:")
                            print(result["output"])
                            
                        if result.get("plot_path"):
                            print(f"\nPlot saved to: {result['plot_path']}")
            except EOFError:
                print("\nReceived EOF. Ending interactive session.")
                self.session_active = False
            except KeyboardInterrupt:
                print("\nReceived keyboard interrupt. Ending interactive session.")
                self.session_active = False
            except Exception as e:
                print(f"\nUnexpected error: {e}")
                print("Continuing session...")
    
    def _process_command(self, command: str):
        """
        Process a command in interactive mode.
        
        Args:
            command: Command string (without the leading '/')
        """
        parts = command.strip().split()
        if not parts:
            return
            
        cmd = parts[0].lower()
        args = parts[1:]
        
        if cmd == "load" and len(args) >= 1:
            file_path = args[0]
            name = args[1] if len(args) > 1 else None
            self.load_csv(file_path, name)
            
        elif cmd == "info":
            df_name = args[0] if args else None
            info = self.get_dataframe_info(df_name)
            if "error" in info:
                print(info["error"])
            else:
                print(f"\nDataset: {info['name']}")
                print(f"Rows: {info['shape']['rows']}, Columns: {info['shape']['columns']}")
                print("\nColumns:")
                for col in info['columns']:
                    print(f"- {col['name']} ({col['dtype']}): {col['unique_values']} unique values, {col['missing_values']} missing")
            
        elif cmd == "list":
            if not self.dataframes:
                print("No datasets loaded")
            else:
                print("Loaded datasets:")
                for name, df in self.dataframes.items():
                    print(f"- {name}: {df.shape[0]} rows, {df.shape[1]} columns")
                    if name == self.current_df_name:
                        print("  (current)")
                        
        elif cmd == "switch" and len(args) >= 1:
            name = args[0]
            if name in self.dataframes:
                self.current_df_name = name
                print(f"Switched to dataset: {name}")
            else:
                print(f"Error: Dataset '{name}' not found")
                
        elif cmd == "clear":
            self.chat_history = []
            print("Chat history cleared")
                
        elif cmd == "help":
            print("Available commands:")
            print("  /load <file_path> [dataset_name] - Load a CSV file")
            print("  /info [dataset_name] - Show dataset information")
            print("  /list - List all loaded datasets")
            print("  /switch <dataset_name> - Switch to a different dataset")
            print("  /clear - Clear chat history")
            print("  /help - Show this help message")
            print("  /exit - Exit the session")
            
        elif cmd == "exit":
            self.session_active = False
            print("Ending interactive session.")
            
        else:
            print(f"Unknown command: {cmd}. Type /help for available commands.")
