#!/bin/bash
# Install frontend dependencies and build
npm install
npm run build

# Move build to where Flask can serve it
rm -rf rag-backend/build
cp -r dist rag-backend/build

# Install Python dependencies
cd rag-backend
pip install -r requirements.txt
