from pydantic import BaseModel, EmailStr, validator
from typing import Optional
from datetime import date, datetime

class UserRegistration(BaseModel):
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    date_of_birth: Optional[date] = None

    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    first_name: str
    last_name: str
    created_at: datetime
    is_active: bool

    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class UserProfileUpdate(BaseModel):
    monthly_income: Optional[float] = None
    savings_goal: Optional[float] = None
    risk_tolerance: Optional[str] = None
    financial_goals: Optional[list[str]] = None

    @validator('risk_tolerance')
    def validate_risk_tolerance(cls, v):
        if v and v not in ['low', 'medium', 'high']:
            raise ValueError('Risk tolerance must be low, medium, or high')
        return v
