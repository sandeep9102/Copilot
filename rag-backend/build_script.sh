#!/bin/bash
# Install frontend dependencies and build
cd frontend
npm install
npm run build
cd ..

# Move build to where Flask can serve it
cp -r frontend/build backend/

# Install Python dependencies
cd backend
pip install -r requirements.txt