import requests
import argparse
import os
import json
from csv_agent import CSVAgent

class OpenWebUIIntegration:
    def __init__(self, openwebui_url="http://localhost:8080", api_key=None):
        """
        Initialize the OpenWebUI integration.
        
        Args:
            openwebui_url: URL for the OpenWebUI server
            api_key: API key for OpenWebUI (if required)
        """
        self.openwebui_url = openwebui_url
        self.api_key = api_key
        self.csv_agent = CSVAgent()
        
    def register_tool(self):
        """
        Register the CSV agent as a tool in OpenWebUI
        """
        # This is a simplified example - actual implementation depends on OpenWebUI's API
        tool_definition = {
            "name": "csv_agent",
            "description": "Analyze and visualize CSV data",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["load", "info", "query", "visualize"],
                        "description": "Action to perform on CSV data"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Parameters for the action"
                    }
                },
                "required": ["action", "parameters"]
            }
        }
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        try:
            response = requests.post(
                f"{self.openwebui_url}/api/tools/register",
                headers=headers,
                json=tool_definition
            )
            if response.status_code == 200:
                print("CSV Agent tool registered successfully!")
                return response.json()
            else:
                print(f"Failed to register tool: {response.status_code} {response.text}")
                return None
        except Exception as e:
            print(f"Error registering tool: {e}")
            return None
    
    def handle_request(self, request_data):
        """
        Handle a request from OpenWebUI.
        
        Args:
            request_data: JSON data from the request
            
        Returns:
            Response data
        """
        try:
            action = request_data.get("action")
            params = request_data.get("parameters", {})
            
            if action == "load":
                file_path = params.get("file_path")
                name = params.get("name")
                result = self.csv_agent.load_csv(file_path, name)
                return {"success": result, "message": f"CSV file loaded: {file_path}"}
                
            elif action == "info":
                df_name = params.get("dataset_name")
                info = self.csv_agent.get_dataframe_info(df_name)
                return info
                
            elif action == "query":
                query = params.get("query")
                df_name = params.get("dataset_name")
                response = self.csv_agent.query_data(query, df_name)
                return {"response": response}
                
            elif action == "visualize":
                viz_type = params.get("type")
                columns = params.get("columns", [])
                df_name = params.get("dataset_name")
                result = self.csv_agent.visualize(viz_type, columns, df_name)
                return result
                
            else:
                return {"error": f"Unknown action: {action}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def start_server(self, port=5000):
        """
        Start a simple HTTP server to handle requests from OpenWebUI.
        
        Args:
            port: Port number for the server
        """
        from flask import Flask, request, jsonify, send_file
        
        app = Flask(__name__)
        
        @app.route('/api/csv_agent', methods=['POST'])
        def handle_request():
            try:
                request_data = request.json
                response = self.handle_request(request_data)
                return jsonify(response)
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @app.route('/api/plots/<path:filename>')
        def serve_plot(filename):
            if os.path.exists(filename):
                return send_file(filename)
            return jsonify({"error": "File not found"}), 404
        
        print(f"Starting CSV Agent server on port {port}")
        app.run(host='0.0.0.0', port=port)

def main():
    parser = argparse.ArgumentParser(description='CSV Agent OpenWebUI Integration')
    parser.add_argument('--port', type=int, default=5000, help='Port for the HTTP server')
    parser.add_argument('--openwebui-url', default='http://localhost:8080', help='OpenWebUI server URL')
    parser.add_argument('--api-key', help='API key for OpenWebUI')
    
    args = parser.parse_args()
    
    integration = OpenWebUIIntegration(
        openwebui_url=args.openwebui_url,
        api_key=args.api_key
    )
    
    # Register the tool with OpenWebUI
    integration.register_tool()
    
    # Start the server
    integration.start_server(port=args.port)

if __name__ == '__main__':
    main()