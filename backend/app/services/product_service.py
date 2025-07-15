from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from app.models.database import Product, ProductCategory, UserPurchase, Transaction
from app.models.products import ProductCreate, ProductPurchase, ProductFilter
from app.services.financial_service import FinancialService
from app.models.financial import TransactionCreate, TransactionCategory
from decimal import Decimal
from typing import List, Optional
import uuid

class ProductService:
    def __init__(self, db: Session):
        self.db = db
        self.financial_service = FinancialService(db)

    def get_all_categories(self) -> List[ProductCategory]:
        """Get all product categories"""
        return self.db.query(ProductCategory).all()

    def get_category_by_id(self, category_id: str) -> Optional[ProductCategory]:
        """Get category by ID"""
        try:
            category_uuid = uuid.UUID(category_id)
            return self.db.query(ProductCategory).filter(ProductCategory.id == category_uuid).first()
        except (ValueError, TypeError):
            return None

    def create_category(self, name: str, description: str, transaction_category: str) -> ProductCategory:
        """Create a new product category"""
        category = ProductCategory(
            name=name,
            description=description,
            transaction_category=transaction_category
        )
        self.db.add(category)
        self.db.commit()
        self.db.refresh(category)
        return category

    def get_products(
        self,
        category_id: Optional[str] = None,
        search: Optional[str] = None,
        min_price: Optional[Decimal] = None,
        max_price: Optional[Decimal] = None,
        limit: int = 20,
        available_only: bool = True
    ) -> List[Product]:
        """Get products with optional filtering"""
        query = self.db.query(Product)

        # Apply filters
        if available_only:
            query = query.filter(Product.is_available == True)

        if category_id:
            try:
                category_uuid = uuid.UUID(category_id)
                query = query.filter(Product.category_id == category_uuid)
            except (ValueError, TypeError):
                pass

        if search:
            search_term = f"%{search}%"
            query = query.filter(
                or_(
                    Product.name.ilike(search_term),
                    Product.description.ilike(search_term),
                    Product.merchant_name.ilike(search_term)
                )
            )

        if min_price is not None:
            query = query.filter(Product.price >= min_price)

        if max_price is not None:
            query = query.filter(Product.price <= max_price)

        return query.limit(limit).all()

    def get_product_by_id(self, product_id: str) -> Optional[Product]:
        """Get product by ID"""
        try:
            product_uuid = uuid.UUID(product_id)
            return self.db.query(Product).filter(Product.id == product_uuid).first()
        except (ValueError, TypeError):
            return None

    def create_product(self, product_data: ProductCreate) -> Product:
        """Create a new product"""
        product = Product(
            name=product_data.name,
            description=product_data.description,
            price=product_data.price,
            category_id=uuid.UUID(product_data.category_id),
            merchant_name=product_data.merchant_name,
            stock_quantity=product_data.stock_quantity,
            image_url=product_data.image_url
        )
        self.db.add(product)
        self.db.commit()
        self.db.refresh(product)
        return product

    def purchase_product(self, user_id: str, product_id: str, quantity: int = 1) -> UserPurchase:
        """Purchase a product and create associated transaction"""
        user_uuid = uuid.UUID(user_id)
        product = self.get_product_by_id(product_id)
        
        if not product:
            raise ValueError("Product not found")
        
        if not product.is_available:
            raise ValueError("Product is not available")

        if product.stock_quantity < quantity:
            raise ValueError("Insufficient stock")

        total_amount = product.price * quantity

        # Get the transaction category from product category
        category = self.get_category_by_id(str(product.category_id))
        if not category:
            raise ValueError("Product category not found")

        # Map transaction category
        transaction_category_map = {
            'groceries': TransactionCategory.GROCERIES,
            'dine_out': TransactionCategory.DINE_OUT,
            'entertainment': TransactionCategory.ENTERTAINMENT,
            'bills': TransactionCategory.BILLS,
            'transport': TransactionCategory.TRANSPORT,
            'shopping': TransactionCategory.SHOPPING,
            'health': TransactionCategory.HEALTH,
            'fitness': TransactionCategory.FITNESS,
            'savings': TransactionCategory.SAVINGS,
            'income': TransactionCategory.INCOME,
            'other': TransactionCategory.OTHER
        }

        transaction_category = transaction_category_map.get(
            category.transaction_category, 
            TransactionCategory.OTHER
        )

        # Create transaction (negative amount for purchase)
        transaction_data = TransactionCreate(
            amount=-total_amount,
            category=transaction_category,
            description=f"Purchase: {product.name} x{quantity}",
            merchant_name=product.merchant_name
        )

        transaction = self.financial_service.process_transaction(user_id, transaction_data)

        # Record purchase
        purchase = UserPurchase(
            user_id=user_uuid,
            product_id=uuid.UUID(product_id),
            transaction_id=transaction.id,
            quantity=quantity,
            total_amount=total_amount
        )

        # Update stock
        product.stock_quantity -= quantity

        self.db.add(purchase)
        self.db.commit()
        self.db.refresh(purchase)
        
        return purchase

    def get_user_purchases(self, user_id: str, limit: int = 20) -> List[UserPurchase]:
        """Get user's purchase history"""
        user_uuid = uuid.UUID(user_id)
        return self.db.query(UserPurchase)\
            .filter(UserPurchase.user_id == user_uuid)\
            .order_by(UserPurchase.purchase_date.desc())\
            .limit(limit)\
            .all()

    def get_purchase_by_id(self, purchase_id: str) -> Optional[UserPurchase]:
        """Get purchase by ID"""
        try:
            purchase_uuid = uuid.UUID(purchase_id)
            return self.db.query(UserPurchase).filter(UserPurchase.id == purchase_uuid).first()
        except (ValueError, TypeError):
            return None

    def update_product_stock(self, product_id: str, new_stock: int) -> Optional[Product]:
        """Update product stock quantity"""
        product = self.get_product_by_id(product_id)
        if product:
            product.stock_quantity = new_stock
            self.db.commit()
            self.db.refresh(product)
        return product

    def toggle_product_availability(self, product_id: str) -> Optional[Product]:
        """Toggle product availability"""
        product = self.get_product_by_id(product_id)
        if product:
            product.is_available = not product.is_available
            self.db.commit()
            self.db.refresh(product)
        return product

    def get_popular_products(self, limit: int = 10, days: int = 30) -> List[Product]:
        """Get most popular products based on purchase history"""
        from datetime import datetime, timedelta
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Query to get products ordered by purchase frequency
        popular_products = self.db.query(Product)\
            .join(UserPurchase)\
            .filter(UserPurchase.purchase_date >= start_date)\
            .group_by(Product.id)\
            .order_by(self.db.func.count(UserPurchase.id).desc())\
            .limit(limit)\
            .all()
        
        return popular_products

    def get_products_by_category(self, category_id: str, limit: int = 20) -> List[Product]:
        """Get products by category ID"""
        try:
            category_uuid = uuid.UUID(category_id)
            return self.db.query(Product)\
                .filter(
                    and_(
                        Product.category_id == category_uuid,
                        Product.is_available == True
                    )
                )\
                .limit(limit)\
                .all()
        except (ValueError, TypeError):
            return []

    def search_products(self, search_term: str, limit: int = 20) -> List[Product]:
        """Search products by name, description, or merchant"""
        search_pattern = f"%{search_term}%"
        return self.db.query(Product)\
            .filter(
                and_(
                    Product.is_available == True,
                    or_(
                        Product.name.ilike(search_pattern),
                        Product.description.ilike(search_pattern),
                        Product.merchant_name.ilike(search_pattern)
                    )
                )
            )\
            .limit(limit)\
            .all()
