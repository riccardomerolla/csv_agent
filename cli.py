import argparse
import os
import cmd
import sys
from csv_agent import CSVAgent

class CSVAgentShell(cmd.Cmd):
    intro = 'Welcome to the CSV Agent shell. Type help or ? to list commands.\n'
    prompt = '(csv_agent) '
    
    def __init__(self, model_name="llama3"):
        super().__init__()
        self.agent = CSVAgent(model_name=model_name)
        
    def do_load(self, arg):
        """Load a CSV file: load <file_path> [dataset_name]"""
        args = arg.split()
        if not args:
            print("Error: File path required")
            return
            
        file_path = args[0]
        name = args[1] if len(args) > 1 else None
        
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found")
            return
            
        result = self.agent.load_csv(file_path, name)
        if result:
            print(f"Successfully loaded {file_path}")
        
    def do_info(self, arg):
        """Show information about the current dataset: info [dataset_name]"""
        name = arg.strip() or None
        info = self.agent.get_dataframe_info(name)
        
        if "error" in info:
            print(info["error"])
            return
            
        print(f"Dataset: {info['name']}")
        print(f"Rows: {info['shape']['rows']}, Columns: {info['shape']['columns']}")
        print("\nColumns:")
        for col in info['columns']:
            print(f"- {col['name']} ({col['dtype']}): {col['unique_values']} unique values, {col['missing_values']} missing")
            
        print("\nSample data (first 5 rows):")
        for i, row in enumerate(info['sample_data'][:5]):
            print(f"{i+1}: {row}")
        
    def do_ask(self, arg):
        """Ask a question about the data: ask <question>"""
        if not arg.strip():
            print("Error: Please provide a question")
            return
            
        response = self.agent.query_data(arg)
        print("\nResponse:")
        print(response)
        
        # Check if there's code to execute
        if "I need to execute code" in response or "```python" in response:
            code_blocks = self._extract_code_blocks(response)
            if code_blocks and self._confirm("Execute the code snippet?"):
                for code in code_blocks:
                    print("\nExecuting code:")
                    print(code)
                    result = self.agent.execute_code(code)
                    
                    if result.get("error"):
                        print(f"Execution error: {result['error']}")
                    elif result.get("output"):
                        print("\nOutput:")
                        print(result["output"])
                        
                    if result.get("plot_path"):
                        print(f"\nPlot saved to: {result['plot_path']}")
    
    def do_viz(self, arg):
        """Create a visualization: viz <type> <column1> [column2...]
        
        Example: viz histogram age
                 viz scatter age income
                 viz bar category
        """
        args = arg.split()
        if len(args) < 2:
            print("Error: Please provide visualization type and at least one column")
            return
            
        viz_type = args[0].lower()
        columns = args[1:]
        
        result = self.agent.visualize(viz_type, columns)
        
        if "error" in result:
            print(f"Visualization error: {result['error']}")
        elif "plot_path" in result:
            print(f"Plot saved to: {result['plot_path']}")
    
    def do_exec(self, arg):
        """Execute Python code on the current DataFrame: exec <code>
        
        The current DataFrame is available as 'df' in the execution context.
        You can use pandas (pd), matplotlib.pyplot (plt), and seaborn (sns).
        
        Example: exec print(df.describe())
        """
        if not arg.strip():
            print("Error: Please provide code to execute")
            return
            
        result = self.agent.execute_code(arg)
        
        if "error" in result:
            print(f"Execution error: {result['error']}")
        elif "output" in result:
            print("\nOutput:")
            print(result["output"])
            
        if "plot_path" in result:
            print(f"Plot saved to: {result['plot_path']}")
    
    def do_list(self, arg):
        """List all loaded datasets"""
        if not self.agent.dataframes:
            print("No datasets loaded")
            return
            
        print("Loaded datasets:")
        for name, df in self.agent.dataframes.items():
            print(f"- {name}: {df.shape[0]} rows, {df.shape[1]} columns")
            if name == self.agent.current_df_name:
                print("  (current)")
    
    def do_switch(self, arg):
        """Switch to a different loaded dataset: switch <dataset_name>"""
        name = arg.strip()
        if not name:
            print("Error: Please provide a dataset name")
            return
            
        if name not in self.agent.dataframes:
            print(f"Error: Dataset '{name}' not found")
            return
            
        self.agent.current_df_name = name
        print(f"Switched to dataset: {name}")
    
    def do_exit(self, arg):
        """Exit the CSV Agent shell"""
        print("Goodbye!")
        return True
        
    def do_quit(self, arg):
        """Exit the CSV Agent shell"""
        return self.do_exit(arg)
        
    def _extract_code_blocks(self, text):
        """Extract code blocks from a markdown-formatted text"""
        import re
        
        # Pattern for code blocks with triple backticks
        pattern = r"```(?:python)?\s*(.*?)\s*```"
        matches = re.findall(pattern, text, re.DOTALL)
        
        if not matches:
            # Try to find inline code
            pattern = r"`(.*?)`"
            matches = re.findall(pattern, text, re.DOTALL)
            
        return matches
    
    def _confirm(self, prompt):
        """Ask for confirmation"""
        response = input(f"{prompt} (y/n) ")
        return response.lower() in ('y', 'yes')
    
    def do_interactive(self, arg):
        """Start an interactive session with the CSV Agent
        
        This enters a REPL mode where you can have a continuous conversation with the agent.
        The agent will maintain context between queries and execute code automatically.
        Type /help within the session for available commands.
        """
        print("Starting interactive session. Type 'exit' to end.")
        self.agent.start_interactive_session()

def main():
    parser = argparse.ArgumentParser(description='CSV Agent CLI')
    parser.add_argument('--model', default='llama3', help='Ollama model name')
    parser.add_argument('--interactive', '-i', action='store_true', help='Start in interactive mode')
    args = parser.parse_args()
    
    shell = CSVAgentShell(model_name=args.model)
    
    if args.interactive:
        shell.do_interactive("")
    else:
        shell.cmdloop()

if __name__ == '__main__':
    main()