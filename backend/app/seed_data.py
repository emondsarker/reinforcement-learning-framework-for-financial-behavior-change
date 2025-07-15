"""
Database seeding script for FinCoach
Populates the database with sample product categories and products
"""

from sqlalchemy.orm import Session
from app.database import SessionLocal, engine
from app.models.database import ProductCategory, Product, Base
from decimal import Decimal
import uuid

def create_sample_categories(db: Session):
    """Create sample product categories"""
    categories = [
        {
            "name": "Grocery Items",
            "description": "Food and household essentials",
            "transaction_category": "groceries"
        },
        {
            "name": "Restaurant Meals",
            "description": "Dining and takeout options",
            "transaction_category": "dine_out"
        },
        {
            "name": "Entertainment",
            "description": "Movies, games, and fun activities",
            "transaction_category": "entertainment"
        },
        {
            "name": "Clothing & Accessories",
            "description": "Apparel and fashion items",
            "transaction_category": "shopping"
        },
        {
            "name": "Electronics",
            "description": "Tech gadgets and devices",
            "transaction_category": "shopping"
        },
        {
            "name": "Health & Wellness",
            "description": "Medical and fitness products",
            "transaction_category": "health"
        },
        {
            "name": "Transportation",
            "description": "Travel and commute options",
            "transaction_category": "transport"
        },
        {
            "name": "Utilities & Services",
            "description": "Bills and essential services",
            "transaction_category": "bills"
        }
    ]
    
    created_categories = {}
    for cat_data in categories:
        category = ProductCategory(**cat_data)
        db.add(category)
        db.commit()
        db.refresh(category)
        created_categories[cat_data["name"]] = category
        print(f"Created category: {category.name}")
    
    return created_categories

def create_sample_products(db: Session, categories: dict):
    """Create sample products for each category"""
    products_data = {
        "Grocery Items": [
            {"name": "Organic Bananas", "price": 3.99, "merchant": "Fresh Market", "description": "Fresh organic bananas, 2 lbs"},
            {"name": "Whole Wheat Bread", "price": 4.50, "merchant": "Bakery Plus", "description": "Artisan whole wheat bread loaf"},
            {"name": "Greek Yogurt", "price": 5.99, "merchant": "Dairy Farm", "description": "Plain Greek yogurt, 32 oz container"},
            {"name": "Chicken Breast", "price": 12.99, "merchant": "Butcher Shop", "description": "Free-range chicken breast, 2 lbs"},
            {"name": "Mixed Vegetables", "price": 6.49, "merchant": "Fresh Market", "description": "Frozen mixed vegetables, 2 lb bag"},
        ],
        "Restaurant Meals": [
            {"name": "Margherita Pizza", "price": 18.99, "merchant": "Tony's Pizzeria", "description": "Classic margherita pizza with fresh basil"},
            {"name": "Chicken Caesar Salad", "price": 14.50, "merchant": "Garden Bistro", "description": "Grilled chicken caesar salad with croutons"},
            {"name": "Beef Burger Combo", "price": 16.99, "merchant": "Burger Palace", "description": "Beef burger with fries and drink"},
            {"name": "Sushi Platter", "price": 24.99, "merchant": "Sakura Sushi", "description": "Assorted sushi and sashimi platter"},
            {"name": "Pasta Carbonara", "price": 19.50, "merchant": "Italian Corner", "description": "Traditional pasta carbonara with pancetta"},
        ],
        "Entertainment": [
            {"name": "Movie Theater Ticket", "price": 12.50, "merchant": "Cinema Complex", "description": "Standard movie ticket"},
            {"name": "Concert Ticket", "price": 75.00, "merchant": "Music Venue", "description": "Live concert general admission"},
            {"name": "Video Game", "price": 59.99, "merchant": "GameStop", "description": "Latest AAA video game title"},
            {"name": "Streaming Subscription", "price": 15.99, "merchant": "StreamFlix", "description": "Monthly streaming service subscription"},
            {"name": "Board Game", "price": 34.99, "merchant": "Toy Store", "description": "Strategy board game for family fun"},
        ],
        "Clothing & Accessories": [
            {"name": "Cotton T-Shirt", "price": 24.99, "merchant": "Fashion Hub", "description": "100% cotton crew neck t-shirt"},
            {"name": "Denim Jeans", "price": 79.99, "merchant": "Denim Co", "description": "Classic fit denim jeans"},
            {"name": "Running Shoes", "price": 129.99, "merchant": "Sports Store", "description": "Lightweight running shoes"},
            {"name": "Leather Wallet", "price": 45.00, "merchant": "Leather Goods", "description": "Genuine leather bifold wallet"},
            {"name": "Winter Jacket", "price": 159.99, "merchant": "Outdoor Gear", "description": "Waterproof winter jacket"},
        ],
        "Electronics": [
            {"name": "Wireless Headphones", "price": 199.99, "merchant": "Tech Store", "description": "Noise-cancelling wireless headphones"},
            {"name": "Smartphone Case", "price": 29.99, "merchant": "Mobile Accessories", "description": "Protective smartphone case"},
            {"name": "Laptop Charger", "price": 79.99, "merchant": "Computer Parts", "description": "Universal laptop charger 65W"},
            {"name": "Bluetooth Speaker", "price": 89.99, "merchant": "Audio Plus", "description": "Portable Bluetooth speaker"},
            {"name": "USB Cable", "price": 19.99, "merchant": "Tech Essentials", "description": "USB-C to USB-A cable 6ft"},
        ],
        "Health & Wellness": [
            {"name": "Multivitamins", "price": 24.99, "merchant": "Health Store", "description": "Daily multivitamin supplement, 90 tablets"},
            {"name": "Protein Powder", "price": 49.99, "merchant": "Fitness Nutrition", "description": "Whey protein powder, 2 lbs"},
            {"name": "Yoga Mat", "price": 39.99, "merchant": "Wellness Shop", "description": "Non-slip yoga mat with carrying strap"},
            {"name": "First Aid Kit", "price": 34.99, "merchant": "Medical Supply", "description": "Complete first aid kit for home"},
            {"name": "Hand Sanitizer", "price": 8.99, "merchant": "Pharmacy", "description": "Antibacterial hand sanitizer, 8 oz"},
        ],
        "Transportation": [
            {"name": "Gas Fill-up", "price": 45.00, "merchant": "Shell Station", "description": "Regular gasoline fill-up"},
            {"name": "Bus Pass", "price": 25.00, "merchant": "City Transit", "description": "Weekly public transit pass"},
            {"name": "Uber Ride", "price": 18.50, "merchant": "Uber", "description": "Ride share service"},
            {"name": "Parking Fee", "price": 12.00, "merchant": "Downtown Parking", "description": "Daily parking fee"},
            {"name": "Car Wash", "price": 15.99, "merchant": "Auto Spa", "description": "Full service car wash"},
        ],
        "Utilities & Services": [
            {"name": "Internet Service", "price": 79.99, "merchant": "ISP Provider", "description": "Monthly high-speed internet"},
            {"name": "Phone Bill", "price": 65.00, "merchant": "Mobile Carrier", "description": "Monthly mobile phone service"},
            {"name": "Electricity Bill", "price": 120.00, "merchant": "Power Company", "description": "Monthly electricity usage"},
            {"name": "Water Bill", "price": 45.00, "merchant": "Water Utility", "description": "Monthly water and sewer service"},
            {"name": "Insurance Premium", "price": 150.00, "merchant": "Insurance Co", "description": "Monthly auto insurance premium"},
        ]
    }
    
    for category_name, products in products_data.items():
        category = categories[category_name]
        for product_data in products:
            product = Product(
                name=product_data["name"],
                description=product_data["description"],
                price=Decimal(str(product_data["price"])),
                category_id=category.id,
                merchant_name=product_data["merchant"],
                is_available=True,
                stock_quantity=100
            )
            db.add(product)
        
        db.commit()
        print(f"Created {len(products)} products for {category_name}")

def seed_database():
    """Main function to seed the database"""
    print("Starting database seeding...")
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Check if data already exists
        existing_categories = db.query(ProductCategory).count()
        if existing_categories > 0:
            print(f"Database already contains {existing_categories} categories. Skipping seeding.")
            return
        
        # Create sample data
        categories = create_sample_categories(db)
        create_sample_products(db, categories)
        
        print("Database seeding completed successfully!")
        
        # Print summary
        total_categories = db.query(ProductCategory).count()
        total_products = db.query(Product).count()
        print(f"Created {total_categories} categories and {total_products} products")
        
    except Exception as e:
        print(f"Error during seeding: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()
