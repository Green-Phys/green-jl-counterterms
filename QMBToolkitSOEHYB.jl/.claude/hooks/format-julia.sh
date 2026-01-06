#!/bin/bash
# PostToolUse hook to format Julia files with JuliaFormatter

# Read JSON input from stdin
input=$(cat)

# Extract file_path from tool_input
file_path=$(echo "$input" | jq -r '.tool_input.file_path // empty')

# Exit if no file path
if [ -z "$file_path" ]; then
    exit 0
fi

# Only format .jl files
if [[ "$file_path" != *.jl ]]; then
    exit 0
fi

# Check if file exists
if [ ! -f "$file_path" ]; then
    exit 0
fi

# Format the file using JuliaFormatter (global installation)
julia -e "using JuliaFormatter; format_file(\"$file_path\")" 2>/dev/null

exit 0
