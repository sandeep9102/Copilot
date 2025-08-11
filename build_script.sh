#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Step 1: Build Frontend
echo "📦 Installing frontend dependencies..."
npm ci  # faster & reproducible installs, uses package-lock.json
npm run build

# Step 2: Move built frontend to Flask
echo "📂 Moving frontend build to Flask backend..."
rm -rf rag-backend/build
cp -r dist rag-backend/build

# Step 3: Install Python backend dependencies
echo "🐍 Installing Python backend dependencies..."
cd rag-backend
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Build completed successfully."
