#!/bin/sh
# usage for converting X (general image file e.g. png) to Y (tiled tiff):
# ./convert_to_tiled_tif.sh X Y
convert $1 -define tiff:tile-geometry=128x128 ptif:$2
