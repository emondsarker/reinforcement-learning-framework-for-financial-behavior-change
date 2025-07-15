# FinCoach Backend API

AI-Powered Financial Wellness Platform with Reinforcement Learning - Backend Implementation

## Overview

The FinCoach backend is a FastAPI-based REST API that provides:

- **User Authentication**: JWT-based authentication with secure password hashing
- **Financial Management**: Virtual wallet system with transaction processing
- **Product Marketplace**: Simulated marketplace for realistic spending scenarios
- **AI Coaching**: CQL-based reinforcement learning model for personalized financial recommendations
- **Analytics**: Comprehensive spending analytics and financial health insights

## Architecture

### Layer Structure

```
backend/
├── app/
│   ├── models/          # Pydantic models and database schemas
│   ├── services/        # Business logic layer
│   ├── routers/         # API endpoints
│   ├── middleware/      # Authentication and other middleware
│   ├── database.py      # Database configuration
│   ├── main.py         # FastAPI application
│   └── seed_data.py    # Database seeding script
├── tests/              # Unit and integration tests
├── requirements.txt    # Python dependencies
├── Dockerfile         # Container configuration
└── .env.example       # Environment variables template
```

### Technology Stack

- **Framework**: FastAPI 0.104.1
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Authentication**: JWT with bcrypt password hashing
- **ML Framework**: PyTorch for CQL model inference
- **Testing**: pytest with test client
- **Containerization**: Docker with multi-stage builds

## Quick Start

### Prerequisites

- Python 3.9+
- PostgreSQL 14+
- Docker (optional)

### Local Development Setup

1. **Clone and navigate to backend directory**

   ```bash
   cd backend
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your database credentials
   ```

5. **Set up PostgreSQL database**

   ```bash
   createdb fincoach_db
   ```

6. **Run database migrations and seed data**

   ```bash
   python app/seed_data.py
   ```

7. **Start the development server**
   ```bash
   python -m uvicorn app.main:app --reload
   ```

The API will be available at `http://localhost:8000`

### Docker Development Setup

1. **Start services with Docker Compose**
   ```bash
   docker-compose up --build
   ```

This will start:

- PostgreSQL database on port 5432
- FastAPI backend on port 8000

## API Documentation

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Main Endpoints

#### Authentication (`/auth`)

- `POST /auth/register` - User registration
- `POST /auth/login` - User login
- `GET /auth/me` - Get current user info
- `POST /auth/refresh` - Refresh JWT token

#### Financial (`/financial`)

- `GET /financial/wallet` - Get user wallet
- `POST /financial/transactions` - Create transaction
- `GET /financial/transactions` - Get transaction history
- `GET /financial/analytics/spending-by-category` - Spending analytics
- `GET /financial/health-summary` - Financial health summary

#### Products (`/products`)

- `GET /products/categories` - Get product categories
- `GET /products/` - Get products with filtering
- `POST /products/purchase` - Purchase a product
- `GET /products/purchases/history` - Purchase history

#### AI Coaching (`/coaching`)

- `GET /coaching/recommendation` - Get AI recommendation
- `GET /coaching/financial-state` - Get financial state analysis
- `GET /coaching/insights` - Get financial insights

## Database Schema

### Core Tables

- **users**: User accounts and authentication
- **user_profiles**: Extended user preferences and goals
- **wallets**: Virtual wallet balances
- **transactions**: Financial transaction records
- **product_categories**: Product categorization
- **products**: Marketplace product catalog
- **user_purchases**: Purchase history and tracking

### Key Relationships

```sql
users (1) -> (1) wallets
users (1) -> (*) transactions
users (1) -> (*) user_purchases
products (*) -> (1) product_categories
user_purchases (*) -> (1) products
user_purchases (*) -> (1) transactions
```

## AI Model Integration

### CQL Model Service

The `CQLModelService` provides:

- **State Vector Generation**: Converts user financial data into ML-ready format
- **Model Inference**: Runs trained CQL model for recommendation generation
- **Fallback Logic**: Rule-based recommendations when model unavailable
- **Health Monitoring**: Model status and performance tracking

### Financial State Vector

17-dimensional state vector including:

- Current balance, weekly spending/income
- Transaction count and velocity
- Category-wise spending breakdown
- Derived metrics (savings rate, spending patterns)

### Coaching Actions

5 possible AI coaching actions: 0. Continue current behavior

1. Spending alert
2. Budget suggestion
3. Savings nudge
4. Positive reinforcement

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_auth.py

# Run with coverage
pytest --cov=app tests/
```

### Test Structure

- **Unit Tests**: Service layer business logic
- **Integration Tests**: API endpoint functionality
- **Authentication Tests**: JWT and user management
- **Financial Tests**: Transaction processing and analytics

## Security Features

### Authentication Security

- JWT tokens with configurable expiration
- bcrypt password hashing with salt
- Secure token verification middleware

### Input Validation

- Pydantic models for request/response validation
- SQL injection prevention with SQLAlchemy ORM
- Type checking and data sanitization

### API Security

- CORS configuration for frontend integration
- Rate limiting ready for production
- Secure headers and error handling

## Performance Considerations

### Database Optimization

- Indexed queries for transaction lookups
- Efficient aggregation for analytics
- Connection pooling with SQLAlchemy

### Caching Strategy

- Model inference caching (ready for Redis)
- Database query optimization
- Static data caching for categories

### Scalability

- Stateless API design
- Horizontal scaling ready
- Database connection management

## Deployment

### Production Environment Variables

```bash
DATABASE_URL=postgresql://user:pass@host:5432/db
JWT_SECRET_KEY=your-production-secret-key
DEBUG=false
LOG_LEVEL=INFO
```

### Docker Production Build

```bash
docker build -t fincoach-backend .
docker run -p 8000:8000 fincoach-backend
```

### Health Checks

- `/health` - Basic service health
- `/coaching/health` - AI model health
- Database connectivity monitoring

## Development Guidelines

### Code Style

- Black code formatting
- Type hints with mypy
- Docstring documentation
- Error handling best practices

### API Design

- RESTful endpoint structure
- Consistent response formats
- Proper HTTP status codes
- Comprehensive error messages

### Database Migrations

- SQLAlchemy model changes
- Alembic migration scripts (future)
- Data seeding and fixtures

## Troubleshooting

### Common Issues

1. **Database Connection Errors**

   - Check PostgreSQL service status
   - Verify DATABASE_URL format
   - Ensure database exists

2. **JWT Token Issues**

   - Check JWT_SECRET_KEY configuration
   - Verify token expiration settings
   - Validate token format

3. **ML Model Loading**
   - Ensure model file exists at MODEL_PATH
   - Check PyTorch compatibility
   - Verify model architecture matches

### Logging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# View application logs
docker-compose logs backend
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit pull request

## License

This project is part of the FinCoach platform for educational and research purposes.
