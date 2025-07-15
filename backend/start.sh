#!/bin/bash

# FinCoach Backend Startup Script

set -e

echo "🚀 Starting FinCoach Backend..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your database credentials"
fi

# Check if PostgreSQL is running
echo "🗄️  Checking database connection..."
if ! pg_isready -h localhost -p 5432 -U fincoach 2>/dev/null; then
    echo "❌ PostgreSQL is not running or not accessible"
    echo "Please start PostgreSQL and ensure the database 'fincoach_db' exists"
    echo "You can create it with: createdb fincoach_db"
    exit 1
fi

# Seed database if needed
echo "🌱 Seeding database..."
python app/seed_data.py

# Start the server
echo "🎯 Starting FastAPI server..."
echo "API will be available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "Press Ctrl+C to stop the server"
echo ""

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
