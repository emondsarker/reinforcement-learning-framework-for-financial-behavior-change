from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.models.auth import UserRegistration, UserLogin, TokenResponse, UserResponse, UserProfileUpdate
from app.services.auth_service import AuthService
from app.middleware.auth_middleware import get_current_active_user
from app.database import get_db
from app.models.database import User

router = APIRouter(prefix="/auth", tags=["authentication"])
auth_service = AuthService()

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegistration, db: Session = Depends(get_db)):
    """Register a new user"""
    try:
        # Register user
        user = auth_service.register_user(user_data, db)
        
        # Create access token
        access_token = auth_service.create_access_token(
            data={"sub": str(user.id)}
        )
        
        # Return token and user info
        return TokenResponse(
            access_token=access_token,
            user=UserResponse(
                id=str(user.id),
                email=user.email,
                first_name=user.first_name,
                last_name=user.last_name,
                created_at=user.created_at,
                is_active=user.is_active
            )
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@router.post("/login", response_model=TokenResponse)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Authenticate user and return access token"""
    try:
        # Authenticate user
        user = auth_service.authenticate_user(credentials, db)
        
        # Create access token
        access_token = auth_service.create_access_token(
            data={"sub": str(user.id)}
        )
        
        # Return token and user info
        return TokenResponse(
            access_token=access_token,
            user=UserResponse(
                id=str(user.id),
                email=user.email,
                first_name=user.first_name,
                last_name=user.last_name,
                created_at=user.created_at,
                is_active=user.is_active
            )
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.post("/logout")
async def logout(current_user: User = Depends(get_current_active_user)):
    """Logout user (client should discard token)"""
    return {"message": "Successfully logged out"}

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get current user information"""
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        created_at=current_user.created_at,
        is_active=current_user.is_active
    )

@router.put("/profile")
async def update_profile(
    profile_data: UserProfileUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update user profile information"""
    try:
        profile = auth_service.update_user_profile(
            str(current_user.id),
            profile_data.dict(exclude_unset=True),
            db
        )
        return {"message": "Profile updated successfully"}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profile update failed"
        )

@router.post("/refresh")
async def refresh_token(current_user: User = Depends(get_current_active_user)):
    """Refresh access token"""
    try:
        # Create new access token
        access_token = auth_service.create_access_token(
            data={"sub": str(current_user.id)}
        )
        
        return TokenResponse(
            access_token=access_token,
            user=UserResponse(
                id=str(current_user.id),
                email=current_user.email,
                first_name=current_user.first_name,
                last_name=current_user.last_name,
                created_at=current_user.created_at,
                is_active=current_user.is_active
            )
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )

@router.delete("/account")
async def deactivate_account(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Deactivate user account"""
    try:
        success = auth_service.deactivate_user(str(current_user.id), db)
        if success:
            return {"message": "Account deactivated successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Account deactivation failed"
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Account deactivation failed"
        )
