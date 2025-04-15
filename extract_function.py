import sys
import ast

def extract_function(file_path, function_name):
    """Extract a function with the given name from a Python file"""
    with open(file_path, 'r') as file:
        file_content = file.read()
        
        #Parse the Python file
        tree = ast.parse(file_content)
        
        #Find the function
        for node in ast.walk(tree):
            if isinstance(node,ast.FunctionDef) and node.name == function_name:
                #Get the source code for the function
                function_lines = file_content.splitlines()[node.lineno-1:node.end_lineno]
                return '\n'.join(function_lines)
    
    return f"# Function '{function_name}' not found in {file_path}"

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_function.py <python_file> <function_name>")
        sys.exit(1)
    
    python_file = sys.argv[1]
    function_name = sys.argv[2]
    
    print(extract_function(python_file, function_name))