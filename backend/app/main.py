from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from app.routers import auth, financial, coaching
from app.database import engine, Base, get_db
from app.middleware.behavioral_tracking_middleware import BehavioralTrackingMiddleware
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables
try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
except Exception as e:
    logger.error(f"Failed to create database tables: {e}")

app = FastAPI(
    title="FinCoach API",
    description="AI-Powered Financial Wellness Platform with Reinforcement Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://localhost:5174",  # Vite dev server (alternate port)
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Behavioral tracking middleware for ML enhancement
app.add_middleware(BehavioralTrackingMiddleware)

# Startup event handler
@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup"""
    try:
        # Initialize continuous learning service
        from app.services.continuous_learning_service import ContinuousLearningService
        from app.services.training_orchestrator import training_orchestrator
        
        db = next(get_db())
        try:
            cl_service = ContinuousLearningService()
            cl_service.initialize_datasets(db)
            logger.info("Continuous learning service initialized successfully")
            
            # Initialize training orchestrator
            logger.info("Training orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing continuous learning service: {e}")
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error during startup: {e}")

# Shutdown event handler
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    try:
        from app.services.training_orchestrator import training_orchestrator
        training_orchestrator.shutdown()
        logger.info("Training orchestrator shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Include routers
app.include_router(auth.router)
app.include_router(financial.router)
app.include_router(coaching.router)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "FinCoach API is running!",
        "version": "1.0.0",
        "description": "AI-Powered Financial Wellness Platform",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "fincoach-api",
        "version": "1.0.0"
    }

@app.get("/info")
async def api_info():
    """API information and available endpoints"""
    return {
        "api_name": "FinCoach API",
        "version": "1.0.0",
        "endpoints": {
            "authentication": "/auth",
            "financial": "/financial",
            "ai_coaching": "/coaching"
        },
        "features": [
            "User authentication with JWT",
            "Financial transaction management",
            "Virtual wallet system",
            "AI-powered financial coaching",
            "Spending analytics and insights"
        ]
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Not Found",
        "message": "The requested resource was not found",
        "status_code": 404
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "status_code": 500
    }

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting FinCoach API on {host}:{port}")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )
