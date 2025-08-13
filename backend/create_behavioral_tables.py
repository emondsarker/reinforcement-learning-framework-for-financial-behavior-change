#!/usr/bin/env python3
"""
Database migration script to create behavioral tracking tables.
Run this script to add BehavioralEvent, UserSegment, PredictionCache, and ModelPerformance tables.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text, Index
from app.database import DATABASE_URL
from app.models.database import Base, BehavioralEvent, UserSegment, PredictionCache, ModelPerformance

def create_behavioral_tables():
    """Create the behavioral tracking tables"""
    
    # Get database URL
    engine = create_engine(DATABASE_URL)
    
    print("Creating behavioral tracking tables...")
    
    try:
        # Create the tables
        BehavioralEvent.__table__.create(engine, checkfirst=True)
        UserSegment.__table__.create(engine, checkfirst=True)
        PredictionCache.__table__.create(engine, checkfirst=True)
        ModelPerformance.__table__.create(engine, checkfirst=True)
        
        # Create indexes for performance optimization
        with engine.connect() as conn:
            # Indexes for BehavioralEvent
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_behavioral_events_user_id 
                ON behavioral_events(user_id);
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_behavioral_events_event_type 
                ON behavioral_events(event_type);
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_behavioral_events_timestamp 
                ON behavioral_events(timestamp);
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_behavioral_events_user_type_time 
                ON behavioral_events(user_id, event_type, timestamp);
            """))
            
            # Indexes for PredictionCache
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_prediction_cache_user_type 
                ON prediction_cache(user_id, prediction_type);
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_prediction_cache_expires_at 
                ON prediction_cache(expires_at);
            """))
            
            # Indexes for ModelPerformance
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_model_performance_name_created 
                ON model_performance(model_name, created_at);
            """))
            
            conn.commit()
        
        print("✅ Successfully created behavioral tracking tables:")
        print("  - behavioral_events")
        print("  - user_segments")
        print("  - prediction_cache")
        print("  - model_performance")
        print("✅ Successfully created performance indexes")
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False
    
    return True

def verify_tables():
    """Verify that the tables were created successfully"""
    
    engine = create_engine(DATABASE_URL)
    
    print("\nVerifying table creation...")
    
    expected_tables = [
        'behavioral_events',
        'user_segments', 
        'prediction_cache',
        'model_performance'
    ]
    
    try:
        with engine.connect() as conn:
            # Check if tables exist
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('behavioral_events', 'user_segments', 'prediction_cache', 'model_performance')
                ORDER BY table_name;
            """))
            
            tables = [row[0] for row in result]
            
            for table in expected_tables:
                if table in tables:
                    print(f"✅ {table} table exists")
                else:
                    print(f"❌ {table} table missing")
            
            # Verify indexes
            print("\nVerifying indexes...")
            index_result = conn.execute(text("""
                SELECT indexname 
                FROM pg_indexes 
                WHERE tablename IN ('behavioral_events', 'user_segments', 'prediction_cache', 'model_performance')
                AND indexname LIKE 'idx_%'
                ORDER BY indexname;
            """))
            
            indexes = [row[0] for row in index_result]
            expected_indexes = [
                'idx_behavioral_events_user_id',
                'idx_behavioral_events_event_type',
                'idx_behavioral_events_timestamp',
                'idx_behavioral_events_user_type_time',
                'idx_prediction_cache_user_type',
                'idx_prediction_cache_expires_at',
                'idx_model_performance_name_created'
            ]
            
            for index in expected_indexes:
                if index in indexes:
                    print(f"✅ {index} index exists")
                else:
                    print(f"❌ {index} index missing")
                    
            return len(tables) == len(expected_tables)
            
    except Exception as e:
        print(f"❌ Error verifying tables: {e}")
        return False

def drop_behavioral_tables():
    """Drop the behavioral tracking tables (for rollback)"""
    
    engine = create_engine(DATABASE_URL)
    
    print("Dropping behavioral tracking tables...")
    
    try:
        with engine.connect() as conn:
            # Drop tables in reverse dependency order
            conn.execute(text("DROP TABLE IF EXISTS model_performance CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS prediction_cache CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS user_segments CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS behavioral_events CASCADE;"))
            conn.commit()
        
        print("✅ Successfully dropped behavioral tracking tables")
        return True
        
    except Exception as e:
        print(f"❌ Error dropping tables: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FinCoach Behavioral Tracking Tables Migration")
    parser.add_argument("--drop", action="store_true", help="Drop tables instead of creating them")
    
    args = parser.parse_args()
    
    print("FinCoach Behavioral Tracking Tables Migration")
    print("=" * 50)
    
    if args.drop:
        # Drop tables
        success = drop_behavioral_tables()
        if success:
            print("\n🗑️  Tables dropped successfully!")
        else:
            print("\n❌ Failed to drop tables")
            sys.exit(1)
    else:
        # Create tables
        success = create_behavioral_tables()
        
        if success:
            # Verify tables
            verify_success = verify_tables()
            
            if verify_success:
                print("\n🎉 Migration completed successfully!")
                print("\nNext steps:")
                print("1. Restart the backend server")
                print("2. Test the enhanced ML service endpoints")
                print("3. Verify behavioral event tracking")
                print("4. Test user segmentation functionality")
            else:
                print("\n⚠️  Migration completed but verification failed")
                sys.exit(1)
        else:
            print("\n❌ Migration failed")
            sys.exit(1)
