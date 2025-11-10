#!/usr/bin/env python3
"""
Renders UK postcode coordinates as a density-based ASCII art map.

This script streams a large CSV file, calculates the geographical bounds,
normalizes the coordinates to a user-defined width (while maintaining aspect ratio),
and generates a text file representing the density of postcodes.

Usage:
  python map_postcodes.py --input ukpostcodes.csv --width 120 --out uk_map.txt
"""

import argparse
import sys
import numpy as np
import pandas as pd

# Characters from light to dense. The first is a space.
DEFAULT_CHARSET = " .:-=+*#%@"

def find_lat_lon_columns(df_columns):
    """
    Finds the latitude and longitude column names from a list of columns.
    Uses case-insensitive and common alternative names.
    """
    cols = {c.lower(): c for c in df_columns}
    lat = cols.get("latitude") or cols.get("lat")
    lon = cols.get("longitude") or cols.get("lon") or cols.get("lng")

    if lat is None:
        lat = next((c for c in df_columns if "lat" in c.lower()), None)
    if lon is None:
        lon = next((c for c in df_columns if "lon" in c.lower() or "lng" in c.lower()), None)
    
    if lat is None or lon is None:
        raise ValueError(f"Could not automatically find latitude/longitude columns in: {df_columns}")
    
    return lat, lon

def get_bounds_and_names(path, chunksize=100_000):
    """
    Streams the CSV to find the min/max bounds and auto-detect column names
    without loading the entire file into memory.
    """
    print(f"Streaming file to determine bounds: {path}")
    it = pd.read_csv(path, chunksize=chunksize, low_memory=False)
    
    min_lat, max_lat = np.inf, -np.inf
    min_lon, max_lon = np.inf, -np.inf
    lat_col, lon_col = None, None

    for i, chunk in enumerate(it):
        if i == 0:
            # Auto-detect column names from the first chunk
            lat_col, lon_col = find_lat_lon_columns(chunk.columns)
            print(f"Detected columns: Latitude='{lat_col}', Longitude='{lon_col}'")

        # Drop rows with invalid data for bounds calculation
        sub = chunk[[lat_col, lon_col]].dropna()
        if sub.empty:
            continue
        
        lats = sub[lat_col].astype(float)
        lons = sub[lon_col].astype(float)
        
        min_lat = min(min_lat, lats.min())
        max_lat = max(max_lat, lats.max())
        min_lon = min(min_lon, lons.min())
        max_lon = max(max_lon, lons.max())

    if min_lat == np.inf:
        raise ValueError("No valid latitude/longitude data found in file.")

    # Add a 2% margin to the bounds for better framing
    lat_margin = (max_lat - min_lat) * 0.02
    lon_margin = (max_lon - min_lon) * 0.02
    
    bounds = (
        min_lat - lat_margin, max_lat + lat_margin,
        min_lon - lon_margin, max_lon + lon_margin
    )
    names = (lat_col, lon_col)
    
    return bounds, names

def rasterize(path, width, height, bounds, names, chunksize=100_000):
    """
    Streams the CSV again, projecting all points onto a 2D density grid.
    """
    print(f"Rasterizing {width}x{height} grid...")
    (min_lat, max_lat, min_lon, max_lon) = bounds
    lat_col, lon_col = names
    
    grid = np.zeros((height, width), dtype=np.int32)
    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon
    
    it = pd.read_csv(path, chunksize=chunksize, low_memory=False)
    
    for chunk in it:
        sub = chunk[[lat_col, lon_col]].dropna()
        if sub.empty:
            continue

        lats = sub[lat_col].astype(float).to_numpy()
        lons = sub[lon_col].astype(float).to_numpy()

        # 1. Normalize coordinates to [0.0, 1.0]
        # (lon - min_lon) / lon_span
        xs_norm = (lons - min_lon) / lon_span
        # (lat - min_lat) / lat_span
        ys_norm = (lats - min_lat) / lat_span
        
        # 2. Scale to grid indices
        # (width - 1) gives indices from 0 to width-1
        ix = (xs_norm * (width - 1)).astype(int)
        
        # 3. Scale and INVERT Y-axis so North (max lat) is at the top (row 0)
        # (1.0 - ys_norm) inverts the y-axis
        iy = ((1.0 - ys_norm) * (height - 1)).astype(int)

        # 4. Filter out any points that fall outside our bounds
        mask = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        ix, iy = ix[mask], iy[mask]

        # 5. Use np.bincount for a very fast way to count points in each cell
        # This converts 2D (iy, ix) indices to a 1D index
        flat_indices = iy * width + ix
        counts = np.bincount(flat_indices, minlength=width * height)
        grid += counts.reshape(height, width)

    print("Rasterization complete.")
    return grid

def grid_to_ascii(grid, charset):
    """
    Converts the numeric density grid into an ASCII string.
    Uses quantiles to create 9 levels of density for the 9 characters.
    """
    # Find all non-zero values to calculate density levels
    vals = grid[grid > 0]
    if vals.size == 0:
        print("Warning: No data points were plotted.")
        return ""

    # Use quantiles to create thresholds. This ensures that the
    # characters are distributed well, even if data is skewed.
    # We create len(charset)-1 boundaries.
    num_levels = len(charset) - 1
    quantiles = np.linspace(0.0, 1.0, num=num_levels + 1)[1:] # e.g., [0.11, 0.22, ... 1.0]
    thresholds = np.quantile(vals, quantiles)
    
    # Ensure thresholds are unique (handles sparse data)
    thresholds = np.unique(thresholds)
    
    # np.digitize maps each grid value to a bin index (0, 1, 2...)
    # This is much faster than looping.
    levels = np.digitize(grid, bins=thresholds, right=True)

    # Convert the 2D array of indices into a list of strings
    lines = []
    for row in levels:
        lines.append("".join(charset[level] for level in row).rstrip())
    
    # Join all rows with newlines
    return "\n".join(lines).rstrip("\n")


def main():
    parser = argparse.ArgumentParser(description="Render UK postcode lat/lon as ASCII art.")
    parser.add_argument("--input", required=True, help="Path to CSV or .zip containing CSV.")
    parser.add_argument("--width", type=int, default=100, help="Width of the ASCII map in characters.")
    parser.add_argument("--out", required=True, help="Output .txt file path.")
    parser.add_argument("--charset", default=DEFAULT_CHARSET, help="String of characters from low to high density.")
    args = parser.parse_args()

    try:
        # --- 1. Get Bounds ---
        # This is a fast first pass over the file to find the geographic extremes
        bounds, (lat_col, lon_col) = get_bounds_and_names(args.input)
        (min_lat, max_lat, min_lon, max_lon) = bounds

        # --- 2. Calculate Aspect Ratio & Height ---
        # We must adjust for the fact that terminal characters are taller than wide
        # A common correction factor is ~0.45 - 0.5
        CHAR_ASPECT_RATIO = 0.45 
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        
        # Calculate height based on width and geographic ratio
        height = int(args.width * (lat_range / lon_range) * CHAR_ASPECT_RATIO)
        
        print(f"Calculated grid size: {args.width} (w) x {height} (h)")

        # --- 3. Rasterize ---
        # This is the second pass, which builds the density grid
        grid = rasterize(args.input, args.width, height, bounds, (lat_col, lon_col))

        # --- 4. Convert to ASCII ---
        art = grid_to_ascii(grid, args.charset)

        # --- 5. Save and Print Preview ---
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(art)

        print(f"\nSuccessfully generated ASCII map at: {args.out}")
        print("\n--- Map Preview (first 25 lines) ---")
        print("\n".join(art.splitlines()[:25]))
        print("...")

    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()