import json
import logging
import time
from typing import Dict, Any, Optional, List
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from app.services.behavioral_event_service import BehavioralEventService
from app.database import get_db
from app.middleware.auth_middleware import get_optional_current_user
import asyncio
import uuid

logger = logging.getLogger(__name__)

class BehavioralTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware for automatic behavioral event tracking"""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.behavioral_service = BehavioralEventService()
        self.tracked_endpoints = {
            # Coaching endpoints
            '/coaching/recommendation': 'recommendation_view',
            '/coaching/feedback': 'recommendation_feedback',
            '/coaching/financial-state': 'financial_state_view',
            '/coaching/insights': 'insights_view',
            '/coaching/history': 'history_view',
            
            # Financial endpoints
            '/financial/transactions': 'transaction_view',
            '/financial/wallet': 'wallet_view',
            '/financial/analytics/spending-by-category': 'analytics_view',
            '/financial/health-summary': 'health_summary_view',
            '/financial/balance-history': 'balance_history_view',
            '/financial/monthly-summary': 'monthly_summary_view',
            
            # Product endpoints
            '/products/': 'product_browse',
            '/products/purchase': 'product_purchase',
            '/products/categories': 'category_browse',
            '/products/popular': 'popular_products_view',
            '/products/search': 'product_search'
        }
        
        # Endpoints that should not be tracked
        self.excluded_endpoints = {
            '/docs', '/redoc', '/openapi.json', '/health', '/info',
            '/auth/login', '/auth/register', '/auth/refresh'
        }

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request and track behavioral events"""
        start_time = time.time()
        
        try:
            # Skip tracking for excluded endpoints
            if self._should_skip_tracking(request):
                return await call_next(request)

            # Extract user information
            user_id = await self._extract_user_id(request)
            
            # Generate session ID for tracking
            session_id = self._generate_session_id(request)
            
            # Track request start
            if user_id:
                await self._track_request_start(request, user_id, session_id)

            # Process the request
            response = await call_next(request)
            
            # Track request completion
            if user_id and response.status_code < 400:
                processing_time = time.time() - start_time
                await self._track_request_completion(
                    request, response, user_id, session_id, processing_time
                )

            return response

        except Exception as e:
            logger.error(f"Error in behavioral tracking middleware: {e}")
            # Continue processing even if tracking fails
            return await call_next(request)

    def _should_skip_tracking(self, request: Request) -> bool:
        """Determine if request should be tracked"""
        path = request.url.path
        
        # Skip excluded endpoints
        for excluded in self.excluded_endpoints:
            if path.startswith(excluded):
                return True
        
        # Skip static files and assets
        if any(path.endswith(ext) for ext in ['.css', '.js', '.png', '.jpg', '.ico', '.svg']):
            return True
            
        # Skip health checks and monitoring
        if 'health' in path or 'metrics' in path:
            return True
            
        return False

    async def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request"""
        try:
            # Try to get user from authorization header
            auth_header = request.headers.get('authorization')
            if not auth_header:
                return None

            # Create a mock database session for user extraction
            db_gen = get_db()
            db = next(db_gen)
            
            try:
                # Use the existing auth middleware function
                from fastapi.security import HTTPAuthorizationCredentials
                from app.middleware.auth_middleware import get_optional_current_user
                
                # Create credentials object
                if auth_header.startswith('Bearer '):
                    token = auth_header.split(' ')[1]
                    credentials = HTTPAuthorizationCredentials(scheme='Bearer', credentials=token)
                    
                    # Get user (this might return None if token is invalid)
                    user = await get_optional_current_user(credentials, db)
                    return str(user.id) if user else None
                    
            finally:
                db.close()
                
        except Exception as e:
            logger.debug(f"Could not extract user ID: {e}")
            return None

    def _generate_session_id(self, request: Request) -> str:
        """Generate or extract session ID"""
        # Try to get session ID from headers or cookies
        session_id = request.headers.get('x-session-id')
        if not session_id:
            session_id = request.cookies.get('session_id')
        
        # Generate new session ID if not found
        if not session_id:
            session_id = str(uuid.uuid4())
            
        return session_id

    async def _track_request_start(self, request: Request, user_id: str, session_id: str):
        """Track the start of a request"""
        try:
            path = request.url.path
            method = request.method
            
            # Determine event type based on endpoint
            event_type = self._get_event_type(path, method)
            
            if event_type:
                # Create a database session for event tracking
                db_gen = get_db()
                db = next(db_gen)
                
                try:
                    event_data = {
                        'page_url': path,
                        'method': method,
                        'session_id': session_id,
                        'interaction_type': 'request_start',
                        'query_params': dict(request.query_params) if request.query_params else {},
                        'user_agent': request.headers.get('user-agent', 'unknown')
                    }
                    
                    await self.behavioral_service.track_event(
                        user_id, event_type, event_data, db
                    )
                    
                finally:
                    db.close()
                    
        except Exception as e:
            logger.error(f"Error tracking request start: {e}")

    async def _track_request_completion(
        self, 
        request: Request, 
        response: Response, 
        user_id: str, 
        session_id: str,
        processing_time: float
    ):
        """Track the completion of a request"""
        try:
            path = request.url.path
            method = request.method
            
            # Determine event type based on endpoint
            event_type = self._get_event_type(path, method)
            
            if event_type:
                # Create a database session for event tracking
                db_gen = get_db()
                db = next(db_gen)
                
                try:
                    event_data = {
                        'page_url': path,
                        'method': method,
                        'session_id': session_id,
                        'interaction_type': 'request_complete',
                        'status_code': response.status_code,
                        'processing_time_ms': round(processing_time * 1000, 2),
                        'response_size': len(response.body) if hasattr(response, 'body') else 0
                    }
                    
                    # Add specific tracking for different endpoint types
                    await self._add_specific_tracking(request, response, event_data, user_id, db)
                    
                    await self.behavioral_service.track_event(
                        user_id, event_type, event_data, db
                    )
                    
                finally:
                    db.close()
                    
        except Exception as e:
            logger.error(f"Error tracking request completion: {e}")

    def _get_event_type(self, path: str, method: str) -> Optional[str]:
        """Determine event type based on path and method"""
        # Check for exact matches first
        if path in self.tracked_endpoints:
            return self.tracked_endpoints[path]
        
        # Check for pattern matches
        if path.startswith('/coaching/'):
            if method == 'GET':
                return 'page_view'
            elif method == 'POST':
                return 'recommendation_interaction'
        elif path.startswith('/financial/'):
            if 'transaction' in path and method == 'POST':
                return 'transaction_creation'
            elif method == 'GET':
                return 'page_view'
        elif path.startswith('/products/'):
            if method == 'POST' and 'purchase' in path:
                return 'purchase'
            elif method == 'GET':
                return 'page_view'
        
        # Default for tracked paths
        if any(path.startswith(prefix) for prefix in ['/coaching/', '/financial/', '/products/']):
            return 'page_view'
            
        return None

    async def _add_specific_tracking(
        self, 
        request: Request, 
        response: Response, 
        event_data: Dict[str, Any], 
        user_id: str,
        db
    ):
        """Add specific tracking for different types of requests"""
        path = request.url.path
        method = request.method
        
        try:
            # Track recommendation interactions
            if path == '/coaching/recommendation' and method == 'GET':
                await self.behavioral_service.track_recommendation_interaction(
                    user_id, 'auto_generated', 'view', event_data.get('session_id'), db
                )
            
            # Track recommendation feedback
            elif path == '/coaching/feedback' and method == 'POST':
                # Try to extract recommendation ID from request body
                if hasattr(request, '_body'):
                    try:
                        body = json.loads(request._body)
                        recommendation_id = body.get('recommendation_id', 'unknown')
                        feedback_type = 'helpful' if body.get('helpful') else 'not_helpful'
                        
                        await self.behavioral_service.track_recommendation_interaction(
                            user_id, recommendation_id, feedback_type, event_data.get('session_id'), db
                        )
                    except (json.JSONDecodeError, AttributeError):
                        pass
            
            # Track product purchases
            elif path == '/products/purchase' and method == 'POST':
                if hasattr(request, '_body'):
                    try:
                        body = json.loads(request._body)
                        product_id = body.get('product_id', 'unknown')
                        quantity = body.get('quantity', 1)
                        
                        await self.behavioral_service.track_purchase_behavior(
                            user_id, product_id, 'purchase', 
                            {'quantity': quantity}, db
                        )
                    except (json.JSONDecodeError, AttributeError):
                        pass
            
            # Track product views
            elif path.startswith('/products/') and method == 'GET':
                # Extract product ID from path if present
                path_parts = path.split('/')
                if len(path_parts) > 2 and path_parts[2] not in ['categories', 'popular', 'search']:
                    product_id = path_parts[2]
                    await self.behavioral_service.track_purchase_behavior(
                        user_id, product_id, 'view', {}, db
                    )
            
            # Track transaction creation
            elif path == '/financial/transactions' and method == 'POST':
                if hasattr(request, '_body'):
                    try:
                        body = json.loads(request._body)
                        transaction_data = {
                            'amount': body.get('amount'),
                            'category': body.get('category'),
                            'transaction_type': body.get('transaction_type')
                        }
                        
                        event_data['transaction_data'] = transaction_data
                    except (json.JSONDecodeError, AttributeError):
                        pass
            
            # Track search behavior
            elif 'search' in path and method == 'GET':
                query_params = dict(request.query_params)
                if 'search' in query_params or 'q' in query_params:
                    search_term = query_params.get('search') or query_params.get('q')
                    event_data['search_term'] = search_term
                    event_data['interaction_type'] = 'search'

        except Exception as e:
            logger.error(f"Error in specific tracking: {e}")

    def _extract_request_body(self, request: Request) -> Optional[Dict]:
        """Safely extract request body"""
        try:
            if hasattr(request, '_body') and request._body:
                return json.loads(request._body)
        except (json.JSONDecodeError, AttributeError):
            pass
        return None

    def _sanitize_tracking_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize tracking data to remove sensitive information"""
        sensitive_keys = ['password', 'token', 'secret', 'key', 'auth', 'credit_card']
        
        sanitized = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = '[REDACTED]'
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_tracking_data(value)
            else:
                sanitized[key] = value
                
        return sanitized

class EventCapture:
    """Helper class for capturing specific events"""
    
    def __init__(self, behavioral_service: BehavioralEventService):
        self.behavioral_service = behavioral_service

    async def capture_page_view(self, user_id: str, page_url: str, session_id: str, db):
        """Capture page view event"""
        await self.behavioral_service.track_page_view(user_id, page_url, session_id, db)

    async def capture_interaction(self, user_id: str, interaction_type: str, data: Dict, db):
        """Capture generic interaction event"""
        await self.behavioral_service.track_event(user_id, 'interaction', data, db)

class PrivacyFilter:
    """Privacy compliance filter for event data"""
    
    @staticmethod
    def filter_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out sensitive data from events"""
        sensitive_patterns = [
            'password', 'pwd', 'secret', 'token', 'key', 'auth',
            'credit', 'card', 'ssn', 'social', 'security', 'pin'
        ]
        
        filtered = {}
        for key, value in data.items():
            if any(pattern in key.lower() for pattern in sensitive_patterns):
                filtered[key] = '[FILTERED]'
            elif isinstance(value, dict):
                filtered[key] = PrivacyFilter.filter_sensitive_data(value)
            elif isinstance(value, str) and len(value) > 500:
                # Truncate very long strings
                filtered[key] = value[:500] + '...'
            else:
                filtered[key] = value
                
        return filtered

class EventBatcher:
    """Batch events for efficient processing"""
    
    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size
        self.events = []
        
    def add_event(self, event: Dict[str, Any]):
        """Add event to batch"""
        self.events.append(event)
        
    def should_process_batch(self) -> bool:
        """Check if batch should be processed"""
        return len(self.events) >= self.batch_size
        
    def get_batch(self) -> List[Dict[str, Any]]:
        """Get current batch and clear it"""
        batch = self.events.copy()
        self.events.clear()
        return batch

class EventValidator:
    """Validate event data before processing"""
    
    @staticmethod
    def validate_event(event_type: str, event_data: Dict[str, Any]) -> bool:
        """Validate event data structure"""
        required_fields = {
            'page_view': ['page_url'],
            'recommendation_interaction': ['recommendation_id', 'interaction_type'],
            'purchase': ['product_id', 'interaction_type'],
            'goal_modification': ['goal_id', 'interaction_type']
        }
        
        if event_type not in required_fields:
            return True  # Allow unknown event types
            
        required = required_fields[event_type]
        return all(field in event_data for field in required)
    
    @staticmethod
    def sanitize_event_data(event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize event data"""
        # Remove None values
        sanitized = {k: v for k, v in event_data.items() if v is not None}
        
        # Ensure string values are not too long
        for key, value in sanitized.items():
            if isinstance(value, str) and len(value) > 1000:
                sanitized[key] = value[:1000] + '...'
                
        return sanitized
