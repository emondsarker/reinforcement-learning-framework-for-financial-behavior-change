#!/usr/bin/env python3
"""
Database migration script to create continuous learning tables.
Run this script to add TrainingDataset, ModelTrainingEvent, ModelVersion, and DataQualityMetrics tables.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text, Index
from app.database import DATABASE_URL
from app.models.database import Base, TrainingDataset, ModelTrainingEvent, ModelVersion, DataQualityMetrics

def create_continuous_learning_tables():
    """Create the continuous learning tables"""
    
    # Get database URL
    engine = create_engine(DATABASE_URL)
    
    print("Creating continuous learning tables...")
    
    try:
        # Create the tables
        TrainingDataset.__table__.create(engine, checkfirst=True)
        ModelTrainingEvent.__table__.create(engine, checkfirst=True)
        ModelVersion.__table__.create(engine, checkfirst=True)
        DataQualityMetrics.__table__.create(engine, checkfirst=True)
        
        # Create indexes for performance optimization
        with engine.connect() as conn:
            # Indexes for TrainingDataset
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_training_datasets_type 
                ON training_datasets(dataset_type);
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_training_datasets_ready 
                ON training_datasets(is_ready_for_training);
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_training_datasets_threshold 
                ON training_datasets(threshold_reached);
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_training_datasets_last_training 
                ON training_datasets(last_training_date);
            """))
            
            # Indexes for ModelTrainingEvent
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_model_training_events_type 
                ON model_training_events(model_type);
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_model_training_events_status 
                ON model_training_events(status);
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_model_training_events_start_time 
                ON model_training_events(start_time);
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_model_training_events_dataset_id 
                ON model_training_events(training_dataset_id);
            """))
            
            # Indexes for ModelVersion
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_model_versions_type 
                ON model_versions(model_type);
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_model_versions_active 
                ON model_versions(is_active);
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_model_versions_deployment_date 
                ON model_versions(deployment_date);
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_model_versions_training_event 
                ON model_versions(training_event_id);
            """))
            
            # Indexes for DataQualityMetrics
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_data_quality_metrics_dataset_id 
                ON data_quality_metrics(training_dataset_id);
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_data_quality_metrics_assessment_date 
                ON data_quality_metrics(assessment_date);
            """))
            
            conn.commit()
        
        print("✅ Successfully created continuous learning tables:")
        print("  - training_datasets")
        print("  - model_training_events")
        print("  - model_versions")
        print("  - data_quality_metrics")
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
        'training_datasets',
        'model_training_events', 
        'model_versions',
        'data_quality_metrics'
    ]
    
    try:
        with engine.connect() as conn:
            # Check if tables exist
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('training_datasets', 'model_training_events', 'model_versions', 'data_quality_metrics')
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
                WHERE tablename IN ('training_datasets', 'model_training_events', 'model_versions', 'data_quality_metrics')
                AND indexname LIKE 'idx_%'
                ORDER BY indexname;
            """))
            
            indexes = [row[0] for row in index_result]
            expected_indexes = [
                'idx_training_datasets_type',
                'idx_training_datasets_ready',
                'idx_training_datasets_threshold',
                'idx_training_datasets_last_training',
                'idx_model_training_events_type',
                'idx_model_training_events_status',
                'idx_model_training_events_start_time',
                'idx_model_training_events_dataset_id',
                'idx_model_versions_type',
                'idx_model_versions_active',
                'idx_model_versions_deployment_date',
                'idx_model_versions_training_event',
                'idx_data_quality_metrics_dataset_id',
                'idx_data_quality_metrics_assessment_date'
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

def drop_continuous_learning_tables():
    """Drop the continuous learning tables (for rollback)"""
    
    engine = create_engine(DATABASE_URL)
    
    print("Dropping continuous learning tables...")
    
    try:
        with engine.connect() as conn:
            # Drop tables in reverse dependency order
            conn.execute(text("DROP TABLE IF EXISTS data_quality_metrics CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS model_versions CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS model_training_events CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS training_datasets CASCADE;"))
            conn.commit()
        
        print("✅ Successfully dropped continuous learning tables")
        return True
        
    except Exception as e:
        print(f"❌ Error dropping tables: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FinCoach Continuous Learning Tables Migration")
    parser.add_argument("--drop", action="store_true", help="Drop tables instead of creating them")
    
    args = parser.parse_args()
    
    print("FinCoach Continuous Learning Tables Migration")
    print("=" * 50)
    
    if args.drop:
        # Drop tables
        success = drop_continuous_learning_tables()
        if success:
            print("\n🗑️  Tables dropped successfully!")
        else:
            print("\n❌ Failed to drop tables")
            sys.exit(1)
    else:
        # Create tables
        success = create_continuous_learning_tables()
        
        if success:
            # Verify tables
            verify_success = verify_tables()
            
            if verify_success:
                print("\n🎉 Migration completed successfully!")
                print("\nNext steps:")
                print("1. Restart the backend server")
                print("2. Test the continuous learning service endpoints")
                print("3. Verify data collection functionality")
                print("4. Test training orchestration")
            else:
                print("\n⚠️  Migration completed but verification failed")
                sys.exit(1)
        else:
            print("\n❌ Migration failed")
            sys.exit(1)
