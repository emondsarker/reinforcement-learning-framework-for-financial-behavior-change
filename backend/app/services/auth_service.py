from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models.database import User, UserProfile
from app.models.auth import UserRegistration, UserLogin
import os
import uuid

class AuthService:
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.secret_key = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return self.pwd_context.verify(plain_password, hashed_password)

    def create_access_token(self, data: dict) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> dict:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError:
            return None

    def register_user(self, user_data: UserRegistration, db: Session) -> User:
        """Register a new user"""
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            raise ValueError("User with this email already exists")

        # Create new user
        hashed_password = self.hash_password(user_data.password)
        db_user = User(
            email=user_data.email,
            password_hash=hashed_password,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            date_of_birth=user_data.date_of_birth
        )
        
        db.add(db_user)
        db.commit()
        db.refresh(db_user)

        # Create default user profile
        user_profile = UserProfile(
            user_id=db_user.id,
            monthly_income=0,
            savings_goal=0,
            risk_tolerance='medium',
            financial_goals='[]'  # Empty JSON array as string
        )
        db.add(user_profile)
        db.commit()

        return db_user

    def authenticate_user(self, credentials: UserLogin, db: Session) -> User:
        """Authenticate a user with email and password"""
        user = db.query(User).filter(User.email == credentials.email).first()
        if not user:
            raise ValueError("Invalid email or password")
        
        if not user.is_active:
            raise ValueError("Account is deactivated")
        
        if not self.verify_password(credentials.password, user.password_hash):
            raise ValueError("Invalid email or password")
        
        return user

    def get_user_by_id(self, user_id: str, db: Session) -> User:
        """Get user by ID"""
        try:
            user_uuid = uuid.UUID(user_id)
            user = db.query(User).filter(User.id == user_uuid).first()
            if not user or not user.is_active:
                return None
            return user
        except (ValueError, TypeError):
            return None

    def update_user_profile(self, user_id: str, profile_data: dict, db: Session) -> UserProfile:
        """Update user profile information"""
        user_uuid = uuid.UUID(user_id)
        profile = db.query(UserProfile).filter(UserProfile.user_id == user_uuid).first()
        
        if not profile:
            # Create profile if it doesn't exist
            profile = UserProfile(user_id=user_uuid)
            db.add(profile)
        
        # Update profile fields
        for field, value in profile_data.items():
            if hasattr(profile, field) and value is not None:
                if field == 'financial_goals':
                    # Convert list to JSON string
                    import json
                    setattr(profile, field, json.dumps(value))
                else:
                    setattr(profile, field, value)
        
        profile.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(profile)
        return profile

    def deactivate_user(self, user_id: str, db: Session) -> bool:
        """Deactivate a user account"""
        try:
            user_uuid = uuid.UUID(user_id)
            user = db.query(User).filter(User.id == user_uuid).first()
            if user:
                user.is_active = False
                user.updated_at = datetime.utcnow()
                db.commit()
                return True
            return False
        except (ValueError, TypeError):
            return False
