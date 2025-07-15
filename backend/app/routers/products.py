from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from app.models.products import (
    ProductResponse, ProductCategoryResponse, ProductPurchase, 
    PurchaseResponse, ProductCreate, CategoryCreate
)
from app.services.product_service import ProductService
from app.middleware.auth_middleware import get_current_active_user, get_optional_current_user
from app.database import get_db
from app.models.database import User

router = APIRouter(prefix="/products", tags=["products"])

@router.get("/categories", response_model=List[ProductCategoryResponse])
async def get_product_categories(db: Session = Depends(get_db)):
    """Get all product categories"""
    try:
        service = ProductService(db)
        categories = service.get_all_categories()
        
        return [
            ProductCategoryResponse(
                id=str(cat.id),
                name=cat.name,
                description=cat.description,
                transaction_category=cat.transaction_category
            )
            for cat in categories
        ]
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve product categories"
        )

@router.post("/categories", response_model=ProductCategoryResponse, status_code=status.HTTP_201_CREATED)
async def create_category(
    category_data: CategoryCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new product category (admin only in future)"""
    try:
        service = ProductService(db)
        category = service.create_category(
            category_data.name,
            category_data.description,
            category_data.transaction_category
        )
        
        return ProductCategoryResponse(
            id=str(category.id),
            name=category.name,
            description=category.description,
            transaction_category=category.transaction_category
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create category"
        )

@router.get("/", response_model=List[ProductResponse])
async def get_products(
    category_id: Optional[str] = Query(None, description="Filter by category ID"),
    search: Optional[str] = Query(None, description="Search products by name, description, or merchant"),
    min_price: Optional[float] = Query(None, description="Minimum price filter"),
    max_price: Optional[float] = Query(None, description="Maximum price filter"),
    limit: int = Query(20, le=100, description="Maximum number of products to return"),
    available_only: bool = Query(True, description="Only return available products"),
    db: Session = Depends(get_db)
):
    """Get products with optional filtering"""
    try:
        service = ProductService(db)
        products = service.get_products(
            category_id=category_id,
            search=search,
            min_price=min_price,
            max_price=max_price,
            limit=limit,
            available_only=available_only
        )
        
        return [
            ProductResponse(
                id=str(p.id),
                name=p.name,
                description=p.description,
                price=p.price,
                category_id=str(p.category_id),
                merchant_name=p.merchant_name,
                is_available=p.is_available,
                stock_quantity=p.stock_quantity,
                image_url=p.image_url
            )
            for p in products
        ]
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve products"
        )

@router.get("/{product_id}", response_model=ProductResponse)
async def get_product_by_id(
    product_id: str,
    db: Session = Depends(get_db)
):
    """Get a specific product by ID"""
    try:
        service = ProductService(db)
        product = service.get_product_by_id(product_id)
        
        if not product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Product not found"
            )
        
        return ProductResponse(
            id=str(product.id),
            name=product.name,
            description=product.description,
            price=product.price,
            category_id=str(product.category_id),
            merchant_name=product.merchant_name,
            is_available=product.is_available,
            stock_quantity=product.stock_quantity,
            image_url=product.image_url
        )
    
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid product ID format"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve product"
        )

@router.post("/", response_model=ProductResponse, status_code=status.HTTP_201_CREATED)
async def create_product(
    product_data: ProductCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new product (admin only in future)"""
    try:
        service = ProductService(db)
        product = service.create_product(product_data)
        
        return ProductResponse(
            id=str(product.id),
            name=product.name,
            description=product.description,
            price=product.price,
            category_id=str(product.category_id),
            merchant_name=product.merchant_name,
            is_available=product.is_available,
            stock_quantity=product.stock_quantity,
            image_url=product.image_url
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create product"
        )

@router.post("/purchase", response_model=PurchaseResponse, status_code=status.HTTP_201_CREATED)
async def purchase_product(
    purchase_data: ProductPurchase,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Purchase a product"""
    try:
        service = ProductService(db)
        purchase = service.purchase_product(
            str(current_user.id),
            purchase_data.product_id,
            purchase_data.quantity
        )
        
        return PurchaseResponse(
            id=str(purchase.id),
            product=ProductResponse(
                id=str(purchase.product.id),
                name=purchase.product.name,
                description=purchase.product.description,
                price=purchase.product.price,
                category_id=str(purchase.product.category_id),
                merchant_name=purchase.product.merchant_name,
                is_available=purchase.product.is_available,
                stock_quantity=purchase.product.stock_quantity,
                image_url=purchase.product.image_url
            ),
            quantity=purchase.quantity,
            total_amount=purchase.total_amount,
            transaction_id=str(purchase.transaction_id),
            purchase_date=purchase.purchase_date
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process purchase"
        )

@router.get("/purchases/history", response_model=List[PurchaseResponse])
async def get_user_purchases(
    limit: int = Query(20, le=100, description="Maximum number of purchases to return"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user's purchase history"""
    try:
        service = ProductService(db)
        purchases = service.get_user_purchases(str(current_user.id), limit)
        
        return [
            PurchaseResponse(
                id=str(p.id),
                product=ProductResponse(
                    id=str(p.product.id),
                    name=p.product.name,
                    description=p.product.description,
                    price=p.product.price,
                    category_id=str(p.product.category_id),
                    merchant_name=p.product.merchant_name,
                    is_available=p.product.is_available,
                    stock_quantity=p.product.stock_quantity,
                    image_url=p.product.image_url
                ),
                quantity=p.quantity,
                total_amount=p.total_amount,
                transaction_id=str(p.transaction_id),
                purchase_date=p.purchase_date
            )
            for p in purchases
        ]
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve purchase history"
        )

@router.get("/popular", response_model=List[ProductResponse])
async def get_popular_products(
    limit: int = Query(10, le=50, description="Maximum number of popular products to return"),
    days: int = Query(30, description="Number of days to consider for popularity"),
    db: Session = Depends(get_db)
):
    """Get most popular products based on purchase history"""
    try:
        service = ProductService(db)
        products = service.get_popular_products(limit, days)
        
        return [
            ProductResponse(
                id=str(p.id),
                name=p.name,
                description=p.description,
                price=p.price,
                category_id=str(p.category_id),
                merchant_name=p.merchant_name,
                is_available=p.is_available,
                stock_quantity=p.stock_quantity,
                image_url=p.image_url
            )
            for p in products
        ]
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve popular products"
        )

@router.get("/category/{category_id}/products", response_model=List[ProductResponse])
async def get_products_by_category(
    category_id: str,
    limit: int = Query(20, le=100, description="Maximum number of products to return"),
    db: Session = Depends(get_db)
):
    """Get products by category ID"""
    try:
        service = ProductService(db)
        products = service.get_products_by_category(category_id, limit)
        
        return [
            ProductResponse(
                id=str(p.id),
                name=p.name,
                description=p.description,
                price=p.price,
                category_id=str(p.category_id),
                merchant_name=p.merchant_name,
                is_available=p.is_available,
                stock_quantity=p.stock_quantity,
                image_url=p.image_url
            )
            for p in products
        ]
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve products by category"
        )

@router.get("/search/{search_term}", response_model=List[ProductResponse])
async def search_products(
    search_term: str,
    limit: int = Query(20, le=100, description="Maximum number of products to return"),
    db: Session = Depends(get_db)
):
    """Search products by name, description, or merchant"""
    try:
        service = ProductService(db)
        products = service.search_products(search_term, limit)
        
        return [
            ProductResponse(
                id=str(p.id),
                name=p.name,
                description=p.description,
                price=p.price,
                category_id=str(p.category_id),
                merchant_name=p.merchant_name,
                is_available=p.is_available,
                stock_quantity=p.stock_quantity,
                image_url=p.image_url
            )
            for p in products
        ]
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search products"
        )
