#!/bin/bash
# Clean orphaned metadata files from review directories

cd /mnt/user/data/CritterCatcher/review || exit 1

echo "Scanning for orphaned metadata files..."
count=0

for dir in */; do
    cd "$dir" || continue
    for json in *.json; do
        # Skip if no json files
        [ -f "$json" ] || continue
        
        # Get corresponding video filename
        video="${json%.json}"
        
        # Check if video exists
        if [ ! -f "$video" ]; then
            echo "Deleting orphaned: $dir$json"
            rm "$json"
            ((count++))
        fi
    done
    cd ..
done

echo "Deleted $count orphaned metadata files"
