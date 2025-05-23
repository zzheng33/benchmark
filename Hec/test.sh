#!/bin/bash

# Loop through directories in the current folder
for dir in */; do
    # Remove trailing slash
    dir=${dir%/}

    # Check if the directory name contains "-cuda"
    if [[ "$dir" == *-cuda* ]]; then
        # Create new name by removing "-cuda"
        new_name="${dir//-cuda/}"

        # Rename the directory
        mv "$dir" "$new_name"
        echo "Renamed: $dir -> $new_name"
    fi
done
