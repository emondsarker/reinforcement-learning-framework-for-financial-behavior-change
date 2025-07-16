#!/usr/bin/env python3
"""
Database migration script to create coaching-related tables.
Run this script to add RecommendationHistory and RecommendationFeedback tables.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text
from app.database import get_database_url
from app.models.database import Base, RecommendationHistory, RecommendationFeedback

def create_coaching_tables():
    """Create the coaching-related tables"""
    
    # Get database URL
    database_url = get_database_url()
    engine = create_engine(database_url)
    
    print("Creating coaching tables...")
    
    try:
        # Create the tables
        RecommendationHistory.__table__.create(engine, checkfirst=True)
        RecommendationFeedback.__table__.create(engine, checkfirst=True)
        
        print("✅ Successfully created coaching tables:")
        print("  - recommendation_history")
        print("  - recommendation_feedback")
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False
    
    return True

def verify_tables():
    """Verify that the tables were created successfully"""
    
    database_url = get_database_url()
    engine = create_engine(database_url)
    
    print("\nVerifying table creation...")
    
    try:
        with engine.connect() as conn:
            # Check if tables exist
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('recommendation_history', 'recommendation_feedback')
                ORDER BY table_name;
            """))
            
            tables = [row[0] for row in result]
            
            if 'recommendation_history' in tables:
                print("✅ recommendation_history table exists")
            else:
                print("❌ recommendation_history table missing")
                
            if 'recommendation_feedback' in tables:
                print("✅ recommendation_feedback table exists")
            else:
                print("❌ recommendation_feedback table missing")
                
            return len(tables) == 2
            
    except Exception as e:
        print(f"❌ Error verifying tables: {e}")
        return False

if __name__ == "__main__":
    print("FinCoach Coaching Tables Migration")
    print("=" * 40)
    
    # Create tables
    success = create_coaching_tables()
    
    if success:
        # Verify tables
        verify_success = verify_tables()
        
        if verify_success:
            print("\n🎉 Migration completed successfully!")
            print("\nNext steps:")
            print("1. Restart the backend server")
            print("2. Test the /coaching/history endpoint")
            print("3. Test the improved /coaching/feedback endpoint")
        else:
            print("\n⚠️  Migration completed but verification failed")
            sys.exit(1)
    else:
        print("\n❌ Migration failed")
        sys.exit(1)
