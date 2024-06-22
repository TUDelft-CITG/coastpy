#!/bin/bash

# Check if a directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

DIRECTORY=$1

# Check if the provided directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Directory does not exist: $DIRECTORY"
    exit 1
fi

# Iterate over all parquet files in the directory
for file in "$DIRECTORY"/*.parquet; do
    if [ -f "$file" ]; then
        # Temporary file for conversion
        tmp_file="${file}.tmp.parquet"

        # Convert the file to GeoParquet
        echo "Converting $file to GeoParquet..."
        gpq convert "$file" "$tmp_file"

        # Move the temporary file back to the original file name
        mv "$tmp_file" "$file"
    fi
done

echo "Conversion completed."
