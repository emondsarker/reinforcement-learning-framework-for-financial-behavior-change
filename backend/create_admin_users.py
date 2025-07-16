#!/usr/bin/env python3
"""
Script to create admin users for FinCoach
Creates admin users with predefined email addresses that have admin access
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy.orm import Session
from app.database import SessionLocal, engine
from app.models.database import User, Wallet, Base
from app.services.auth_service import AuthService
import uuid

def create_admin_users():
    """Create admin users with predefined admin emails"""
    
    print("Creating admin users...")
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Admin users to create
        admin_users = [
            {
                "email": "admin@fincoach.com",
                "password": "password123",
                "first_name": "Admin",
                "last_name": "User"
            },
            {
                "email": "admin@example.com", 
                "password": "password123",
                "first_name": "System",
                "last_name": "Administrator"
            },
            {
                "email": "test@admin.com",
                "password": "password123", 
                "first_name": "Test",
                "last_name": "Admin"
            }
        ]
        
        created_count = 0
        
        for admin_data in admin_users:
            # Check if user already exists
            existing_user = db.query(User).filter(User.email == admin_data["email"]).first()
            
            if existing_user:
                print(f"✅ Admin user {admin_data['email']} already exists")
                continue
            
            # Hash the password
            password_hash = AuthService.hash_password(admin_data["password"])
            
            # Create the user
            admin_user = User(
                email=admin_data["email"],
                password_hash=password_hash,
                first_name=admin_data["first_name"],
                last_name=admin_data["last_name"],
                is_active=True
            )
            
            db.add(admin_user)
            db.flush()  # Get the user ID
            
            # Create a wallet for the admin user
            admin_wallet = Wallet(
                user_id=admin_user.id,
                balance=10000.00,  # Give admin users $10,000 starting balance
                currency="USD"
            )
            
            db.add(admin_wallet)
            db.commit()
            
            print(f"✅ Created admin user: {admin_data['email']}")
            created_count += 1
        
        if created_count > 0:
            print(f"\n🎉 Successfully created {created_count} admin users!")
        else:
            print("\n✅ All admin users already exist!")
            
        print("\nAdmin Login Credentials:")
        print("=" * 40)
        for admin_data in admin_users:
            print(f"Email: {admin_data['email']}")
            print(f"Password: {admin_data['password']}")
            print("-" * 40)
        
        print("\nTo access admin panel:")
        print("1. Login with any of the above credentials")
        print("2. Navigate to /admin in your browser")
        print("3. Admin access is determined by email address")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating admin users: {e}")
        db.rollback()
        return False
    finally:
        db.close()

def verify_admin_users():
    """Verify that admin users were created successfully"""
    
    print("\nVerifying admin users...")
    
    db = SessionLocal()
    
    try:
        admin_emails = [
            "admin@fincoach.com",
            "admin@example.com", 
            "test@admin.com"
        ]
        
        for email in admin_emails:
            user = db.query(User).filter(User.email == email).first()
            if user:
                wallet = db.query(Wallet).filter(Wallet.user_id == user.id).first()
                print(f"✅ {email} - User ID: {user.id}, Wallet Balance: ${wallet.balance if wallet else 'No wallet'}")
            else:
                print(f"❌ {email} - User not found")
                
        return True
        
    except Exception as e:
        print(f"❌ Error verifying admin users: {e}")
        return False
    finally:
        db.close()

if __name__ == "__main__":
    print("FinCoach Admin User Creation")
    print("=" * 40)
    
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    
    # Create admin users
    success = create_admin_users()
    
    if success:
        # Verify admin users
        verify_success = verify_admin_users()
        
        if verify_success:
            print("\n🎉 Admin user creation completed successfully!")
            print("\nNext steps:")
            print("1. Start/restart your frontend and backend servers")
            print("2. Go to the login page")
            print("3. Login with any admin email and password: password123")
            print("4. Navigate to /admin to access the admin panel")
        else:
            print("\n⚠️  Admin users created but verification failed")
            sys.exit(1)
    else:
        print("\n❌ Admin user creation failed")
        sys.exit(1)
