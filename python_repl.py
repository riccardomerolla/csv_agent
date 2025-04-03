import sys
import io
import contextlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid
import traceback
from typing import Dict, Any, Optional

class PythonREPL:
    """
    A Python REPL (Read-Eval-Print Loop) that executes code in a controlled environment.
    Provides isolation and captures outputs for safe code execution.
    """
    
    def __init__(self, variables: Optional[Dict[str, Any]] = None):
        """
        Initialize the Python REPL with optional variables.
        
        Args:
            variables: Dictionary of variables to include in the execution environment
        """
        self.locals = {}
        
        # Set up default imports and variables
        default_vars = {
            'pd': pd,
            'plt': plt,
            'sns': sns,
        }
        
        # Combine default with provided variables
        if variables:
            default_vars.update(variables)
            
        self.locals.update(default_vars)
    
    def execute(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code and return the results.
        
        Args:
            code: Python code to execute
            
        Returns:
            Dictionary containing execution results:
                - output: stdout content
                - error: error message if execution failed
                - plot_path: path to saved plot if a plot was generated
        """
        # Prepare result container
        result = {
            "output": None,
            "error": None,
            "plot_path": None
        }
        
        # Set up output capture
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Check if the code includes plotting commands
        has_plot = any(plot_cmd in code for plot_cmd in ['plt.', 'plot(', 'savefig', 'imshow', 'sns.'])
        
        # Generate a unique plot path if needed
        plot_path = None
        if has_plot:
            plot_id = str(uuid.uuid4())[:8]
            plot_path = f"plot_{plot_id}.png"
            
            # Add plot saving code if not already present
            if 'plt.savefig' not in code and 'savefig' not in code:
                save_code = f"\nplt.savefig('{plot_path}')\nplt.close()"
                code += save_code

        try:
            # Capture stdout and stderr during execution
            with contextlib.redirect_stdout(stdout_capture):
                with contextlib.redirect_stderr(stderr_capture):
                    exec(code, globals(), self.locals)
            
            # Check if plot was saved
            if has_plot and os.path.exists(plot_path):
                result["plot_path"] = plot_path
                
            # Get captured output
            stdout_output = stdout_capture.getvalue().strip()
            if stdout_output:
                result["output"] = stdout_output
                
            # Check for any result stored in the '_result' variable
            if '_result' in self.locals:
                if result["output"]:
                    result["output"] += f"\n{str(self.locals['_result'])}"
                else:
                    result["output"] = str(self.locals['_result'])
                    
        except Exception as e:
            # Capture traceback for detailed error reporting
            error_tb = traceback.format_exc()
            result["error"] = f"{str(e)}\n{error_tb}"
            
            # Also capture any stderr output
            stderr_output = stderr_capture.getvalue().strip()
            if stderr_output:
                result["error"] = f"{result['error']}\n{stderr_output}"
        
        return result
    
    def get_variable(self, name: str) -> Any:
        """
        Get a variable from the REPL's namespace.
        
        Args:
            name: Variable name
            
        Returns:
            The variable value or None if not found
        """
        return self.locals.get(name)
    
    def set_variable(self, name: str, value: Any) -> None:
        """
        Set a variable in the REPL's namespace.
        
        Args:
            name: Variable name
            value: Variable value
        """
        self.locals[name] = value
