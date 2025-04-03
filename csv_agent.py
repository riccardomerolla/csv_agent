import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import re
from typing import Dict, List, Optional, Union, Any
import requests
from python_repl import PythonREPL

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
        self.repl = None
        
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
            if self.repl is None:
                self.repl = PythonREPL(variables={'df': df})
            else:
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
    
    def query_data(self, query: str, df_name: Optional[str] = None) -> str:
        """
        Process a natural language query about the data using Ollama.
        
        Args:
            query: Natural language query from the user
            df_name: Name of the DataFrame to query
            
        Returns:
            Response from the model
        """
        df_name = df_name or self.current_df_name
        if df_name not in self.dataframes:
            return f"Error: DataFrame '{df_name}' not found"
        
        # Update REPL with the current dataframe
        self.repl.set_variable('df', self.dataframes[df_name])
        
        df_info = self.get_dataframe_info(df_name)
        
        # Construct a prompt for the LLM
        prompt = f"""
You are an agent that writes and executes python code
You have access to a Python, which you can use to execute the python code.
You must write the python code code assuming that the dataframe (stored as df) has already been read.
If you get an error, debug your code and try again.
You might know the answer without running any code, but you should still run the code to get the answer.
If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
Do not create example dataframes
Answer the following query about the dataset:

{query}

Here is information about the dataset:
- Name: {df_info['name']}
- Shape: {df_info['shape']['rows']} rows and {df_info['shape']['columns']} columns
- Columns: {', '.join([c['name'] for c in df_info['columns']])}

Here are details about each column:
{json.dumps([{c['name']: c} for c in df_info['columns']], indent=2)}

Sample data (first 5 rows):
{json.dumps(df_info['sample_data'], indent=2)}

If the user asks for Python code, provide executable code using pandas. If they want a visualization, provide matplotlib/seaborn code.
If you need to execute code to answer this question precisely, say "I need to execute code to answer this accurately" and then provide the code. 
"""
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
        
        # Initialize REPL if it doesn't exist yet
        if self.repl is None:
            self.repl = PythonREPL(variables={'df': self.dataframes[df_name]})
        else:
            # Update the dataframe in the REPL
            self.repl.set_variable('df', self.dataframes[df_name])
        
        # Execute the code using the REPL
        result = self.repl.execute(code)
        
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
        
        # Initialize REPL if it doesn't exist yet
        if self.repl is None:
            self.repl = PythonREPL(variables={'df': df})
        else:
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
                code += f"plt.scatter(df['{columns[0]}'], df['{columns[1]}'])\n"
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
                    code += f"plt.title('Line plot of {', '.join(columns[1:])} vs {x_col}')\n"
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
                "temperature": 0.7,
                "max_tokens": self.max_tokens
            }
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            return f"Error calling Ollama API: {e}"
