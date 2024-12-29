#!/bin/bash

# Set paths based on your project structure
PROJECT_DIR="/scratch/user/aaupadhy/college/RA/FlowNet"
SHAPEFILE_DIR="$PROJECT_DIR/outputs/cartopy_shapefiles"
SHAPEFILE_URL="https://naturalearth.s3.amazonaws.com/50m_physical/ne_50m_coastline.zip"

# Ensure the shapefile directory exists
mkdir -p "$SHAPEFILE_DIR"

# Download the shapefiles
wget -O "$PROJECT_DIR/ne_50m_coastline.zip" "$SHAPEFILE_URL"

# Extract the files to the shapefile directory
unzip -o "$PROJECT_DIR/ne_50m_coastline.zip" -d "$SHAPEFILE_DIR"
rm "$PROJECT_DIR/ne_50m_coastline.zip"  # Clean up zip file

# Verify the shapefile setup
if [[ -f "$SHAPEFILE_DIR/ne_50m_coastline.shp" ]]; then
    echo "Shapefiles successfully set up in $SHAPEFILE_DIR"
else
    echo "Failed to set up shapefiles. Check manually."
    exit 1
fi
