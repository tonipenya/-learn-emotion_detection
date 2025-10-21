#!/usr/bin/env bash
set -e

rm -rf app
mkdir app

echo "Copying assets to app folder"
cp static/* app/
cp data/model.onnx app/
