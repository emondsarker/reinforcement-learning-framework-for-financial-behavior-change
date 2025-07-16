# FinCoach Architecture Overview

**Last Updated**: January 15, 2025  
**Current Status**: Task 12 Complete - Transaction Management & Wallet Page Implemented

## 🏗️ System Architecture

### High-Level Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   Database      │
│   (React TS)    │◄──►│   (FastAPI)     │◄──►│  (PostgreSQL)   │
│                 │    │                 │    │                 │
│ • Auth Context  │    │ • JWT Auth      │    │ • User Data     │
│ • TanStack Query│    │ • REST APIs     │    │ • Transactions  │
│ • Tailwind CSS │    │ • ML Integration│    │ • Products      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   ML Models     │
                       │   (PyTorch)     │
                       │                 │
                       │ • CQL Model     │
                       │ • Inference     │
                       │ • Recommendations│
                       └─────────────────┘
```

## 📂 Current Implementation Status

### ✅ Completed Components

#### Backend (FastAPI)

- **Authentication System**: JWT-based auth with user registration/login
- **Database Layer**: SQLAlchemy ORM with PostgreSQL
- **API Endpoints**: RESTful APIs for auth, financial, products, coaching
- **Middleware**: CORS, authentication middleware
- **Services**: Business logic separation
- **Models**: Pydantic models for request/response validation
- **Docker Support**: Containerized deployment

#### Frontend (React TypeScript)

- **Project Foundation**: Vite + React + TypeScript + Tailwind CSS
- **API Client**: Axios with JWT interceptors and error handling
- **State Management**: TanStack Query for server state + React Context for auth
- **Type System**: Complete TypeScript interfaces matching backend
- **Authentication**: JWT token management, auth context, and hooks
- **UI Framework**: Tailwind CSS with custom black & white theme
- **Development Tools**: ESLint, Prettier, hot reload

#### ML Components

- **Data Processing**: Comprehensive transaction data cleaning and analysis
- **RL Framework**: State, action, reward definitions for financial coaching
- **CQL Model**: Trained Conservative Q-Learning model for safe recommendations
- **Model Artifacts**: Serialized model ready for backend integration

### 🚧 In Progress

#### Frontend UI Development

- **Next Task**: Task 13 - Analytics & Insights Page
- **Recently Completed**: Task 12 - Transaction Management & Wallet Page with comprehensive transaction interface

### ✅ Recently Completed (Task 12)

#### Transaction Management & Wallet Page

- **Wallet Components**: Complete set of wallet management components with advanced functionality
- **Transaction History**: Paginated transaction list with comprehensive filtering and search
- **Balance Visualization**: 30-day balance trend chart using Recharts with interactive tooltips
- **Transaction Creation**: Modal form for manual transaction creation with validation
- **Advanced Filtering**: Date range, categories, transaction type, amount range, and search filters
- **Mobile Responsive**: Full responsive design optimized for all device sizes
- **Integration**: Seamless integration with existing financial hooks and backend APIs

## 🔧 Technical Stack

### Frontend Technologies

```typescript
{
  "framework": "React 18 + TypeScript",
  "build": "Vite",
  "styling": "Tailwind CSS v3",
  "state": "TanStack Query + React Context",
  "http": "Axios",
  "validation": "Zod",
  "forms": "React Hook Form",
  "routing": "React Router (planned)",
  "ui": "@headlessui/react + @heroicons/react"
}
```

### Backend Technologies

```python
{
    "framework": "FastAPI",
    "database": "PostgreSQL + SQLAlchemy",
    "auth": "JWT (python-jose)",
    "validation": "Pydantic",
    "ml": "PyTorch + scikit-learn",
    "deployment": "Docker + uvicorn",
    "testing": "pytest"
}
```

## 🔐 Authentication Flow

### Current Implementation

1. **Registration/Login**: Frontend sends credentials to `/auth/register` or `/auth/login`
2. **Token Generation**: Backend validates and returns JWT token + user data
3. **Token Storage**: Frontend stores token in localStorage
4. **Request Interceptor**: Axios automatically adds `Authorization: Bearer <token>` header
5. **Token Validation**: Backend middleware validates token on protected routes
6. **Auto-Logout**: Frontend detects expired tokens and clears auth state

### Security Features

- JWT token expiration checking
- Automatic logout on token expiry
- CORS configuration for cross-origin requests
- Password validation (8+ chars, uppercase, digit)
- Protected route middleware

## 📊 Data Flow Architecture

### Authentication Data Flow

```
Frontend Auth Context ←→ AuthService ←→ Backend API ←→ Database
                    ↓
              TanStack Query
                    ↓
              Component State
```

### API Communication Pattern

```
React Component → useAuth Hook → TanStack Query → AuthService → Axios → Backend API
                                      ↓
                              Cache Management
                                      ↓
                              UI State Updates
```

## 🗂️ File Structure Overview

### Frontend Architecture

```
frontend/src/
├── components/          # React components (organized by feature)
├── contexts/           # React contexts (AuthContext)
├── hooks/              # Custom hooks (useAuth, useCurrentUser)
├── services/           # API services (authService, api client)
├── types/              # TypeScript type definitions
├── lib/                # Utility libraries (queryClient, queryKeys)
├── pages/              # Page components (planned)
├── utils/              # Utility functions (planned)
└── App.tsx             # Main application component
```

### Backend Architecture

```
backend/app/
├── models/             # SQLAlchemy & Pydantic models
├── services/           # Business logic services
├── routers/            # API route handlers
├── middleware/         # Custom middleware
├── database.py         # Database configuration
└── main.py             # FastAPI application entry
```

## 🔄 Development Workflow

### Current Process

1. **Backend-First**: API endpoints implemented first
2. **Type Generation**: TypeScript types match backend models
3. **Service Layer**: API services abstract HTTP calls
4. **Hook Layer**: TanStack Query hooks manage server state
5. **Component Layer**: React components consume hooks
6. **Testing**: Manual testing with running backend

### Next Steps (Task 5+)

1. **Routing Setup**: React Router with protected routes
2. **Layout Components**: Header, navigation, layout structure
3. **Form Components**: Login/register forms with validation
4. **Dashboard**: Wallet display and transaction management
5. **Marketplace**: Product browsing and purchase interface

## 🚀 Deployment Architecture

### Current Setup

- **Development**: Local development with hot reload
- **Backend**: Docker container with PostgreSQL
- **Frontend**: Vite dev server
- **Database**: PostgreSQL in Docker container

### Production Ready Features

- Docker Compose orchestration
- Environment variable configuration
- CORS properly configured
- Database migrations ready
- Health check endpoints

## 📈 Performance Considerations

### Frontend Optimizations

- **TanStack Query Caching**: Smart cache invalidation and background updates
- **Code Splitting**: Vite's automatic code splitting
- **Type Safety**: Compile-time error catching
- **Bundle Size**: Tree shaking and modern build tools

### Backend Optimizations

- **Database Indexing**: Proper indexes on user_id, email, timestamps
- **Connection Pooling**: SQLAlchemy connection management
- **Async Operations**: FastAPI async/await pattern
- **Response Caching**: Planned for static data

## 🔮 Future Architecture Plans

### Immediate (Tasks 5-8)

- React Router integration
- Form validation with Zod schemas
- Dashboard with real-time data
- Responsive design implementation

### Medium Term (Tasks 13-15)

- Analytics dashboard with spending insights
- Admin panel for system management
- Final polish and error handling

### Long Term (Future Enhancements)

- Advanced analytics and reporting
- Performance optimizations
- Enhanced mobile features
- Production deployment pipeline

## 🧪 Testing Strategy

### Current Testing

- Manual API testing via FastAPI docs
- Frontend component testing in browser
- Authentication flow validation

### Planned Testing

- Unit tests for services and hooks
- Integration tests for API endpoints
- E2E tests for critical user flows
- Performance testing for ML inference

---

**Note**: This architecture document should be updated as new components are implemented. Each completed task should update the relevant sections to maintain accuracy for future development.
