#!/usr/bin/env python3
"""
Complete setup script for FinCoach
Creates all necessary database tables and admin users
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from app.database import SessionLocal, engine, DATABASE_URL
from app.models.database import Base, RecommendationHistory, RecommendationFeedback, User, Wallet
from app.services.auth_service import AuthService

def create_all_tables():
    """Create all database tables"""
    
    print("Creating all database tables...")
    
    try:
        # Create all tables from models
        Base.metadata.create_all(bind=engine)
        
        print("✅ Successfully created all database tables")
        return True
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False

def verify_tables():
    """Verify that all tables were created successfully"""
    
    print("\nVerifying table creation...")
    
    try:
        with engine.connect() as conn:
            # Check if key tables exist
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN (
                    'users', 'wallets', 'transactions', 'products', 
                    'product_categories', 'user_purchases',
                    'recommendation_history', 'recommendation_feedback'
                )
                ORDER BY table_name;
            """))
            
            tables = [row[0] for row in result]
            
            expected_tables = [
                'users', 'wallets', 'transactions', 'products',
                'product_categories', 'user_purchases', 
                'recommendation_history', 'recommendation_feedback'
            ]
            
            for table in expected_tables:
                if table in tables:
                    print(f"✅ {table} table exists")
                else:
                    print(f"❌ {table} table missing")
                    
            return len(tables) >= 6  # At least core tables should exist
            
    except Exception as e:
        print(f"❌ Error verifying tables: {e}")
        return False

def create_admin_users():
    """Create admin users with predefined admin emails"""
    
    print("\nCreating admin users...")
    
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
            auth_service = AuthService()
            password_hash = auth_service.hash_password(admin_data["password"])
            
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
        
        verified_count = 0
        
        for email in admin_emails:
            user = db.query(User).filter(User.email == email).first()
            if user:
                wallet = db.query(Wallet).filter(Wallet.user_id == user.id).first()
                print(f"✅ {email} - User ID: {user.id}, Wallet Balance: ${wallet.balance if wallet else 'No wallet'}")
                verified_count += 1
            else:
                print(f"❌ {email} - User not found")
                
        return verified_count >= 1  # At least one admin should exist
        
    except Exception as e:
        print(f"❌ Error verifying admin users: {e}")
        return False
    finally:
        db.close()

def run_seed_data():
    """Run the seed data script to create sample products"""
    
    print("\nRunning seed data script...")
    
    try:
        # Import and run seed data
        from app.seed_data import seed_database
        seed_database()
        print("✅ Seed data completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error running seed data: {e}")
        return False

def main():
    """Main setup function"""
    
    print("FinCoach Complete Setup")
    print("=" * 50)
    
    success_steps = 0
    total_steps = 5
    
    # Step 1: Create all tables
    if create_all_tables():
        success_steps += 1
    
    # Step 2: Verify tables
    if verify_tables():
        success_steps += 1
    
    # Step 3: Create admin users
    if create_admin_users():
        success_steps += 1
    
    # Step 4: Verify admin users
    if verify_admin_users():
        success_steps += 1
    
    # Step 5: Run seed data
    if run_seed_data():
        success_steps += 1
    
    print(f"\nSetup completed: {success_steps}/{total_steps} steps successful")
    
    if success_steps == total_steps:
        print("\n🎉 FinCoach setup completed successfully!")
        print("\nAdmin Login Credentials:")
        print("=" * 40)
        print("Email: admin@fincoach.com")
        print("Password: password123")
        print("-" * 40)
        print("Email: admin@example.com")
        print("Password: password123")
        print("-" * 40)
        print("Email: test@admin.com")
        print("Password: password123")
        print("=" * 40)
        
        print("\nNext steps:")
        print("1. Start/restart your backend server")
        print("2. Start/restart your frontend server")
        print("3. Go to the login page")
        print("4. Login with any admin email and password: password123")
        print("5. Navigate to /admin to access the admin panel")
        print("6. Test the /coaching/history and /coaching/feedback endpoints")
        
        return True
    else:
        print(f"\n⚠️  Setup completed with {total_steps - success_steps} errors")
        print("Please check the error messages above and try again")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
