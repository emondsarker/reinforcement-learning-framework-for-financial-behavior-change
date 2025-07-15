from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.database import User
from app.services.auth_service import AuthService
from typing import Optional

security = HTTPBearer()
auth_service = AuthService()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user from JWT token"""
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Verify JWT token
        payload = auth_service.verify_token(credentials.credentials)
        if payload is None:
            raise credentials_exception

        # Extract user ID from token
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception

        # Get user from database
        user = auth_service.get_user_by_id(user_id, db)
        if user is None:
            raise credentials_exception

        return user

    except Exception:
        raise credentials_exception

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user (additional check for user status)"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

async def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Get current user if authenticated, otherwise return None"""
    
    if credentials is None:
        return None

    try:
        # Verify JWT token
        payload = auth_service.verify_token(credentials.credentials)
        if payload is None:
            return None

        # Extract user ID from token
        user_id: str = payload.get("sub")
        if user_id is None:
            return None

        # Get user from database
        user = auth_service.get_user_by_id(user_id, db)
        return user

    except Exception:
        return None

class RoleChecker:
    """Role-based access control (for future expansion)"""
    
    def __init__(self, allowed_roles: list[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, current_user: User = Depends(get_current_active_user)):
        # For now, all authenticated users have the same role
        # This can be expanded when user roles are implemented
        return current_user

# Convenience dependency for admin access (future use)
admin_required = RoleChecker(["admin"])
user_required = RoleChecker(["user", "admin"])
