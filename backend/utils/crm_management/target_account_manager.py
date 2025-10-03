from backend.database import get_db, Base
from backend.utils.crm_management.target_account_models import TargetAccount
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, Integer, JSON
import uuid

"""
For the list of account data provided this function looks in the target_list_db if an existing record is found, else creates one.
1. For this we need to first create the target_list_db in the database
DB has columns: Product_ID | Target Account ID (uuid generated) | Target Account | Revenue | Employees | Funding | Industry | Geography
2. For the given product ID check if the target account already exists - else add to DB

"""

class TargetAccount(Base):
    __tablename__ = "target_accounts"

    id = Column(String, primary_key=True, index=True)
    product_id = Column(String, index=True)
    account_name = Column(String, index=True)
    industry = Column(String)
    revenue_range = Column(String)
    employee_range = Column(String)
    funding_stage = Column(String)
    geography = Column(String)
    competitor_used = Column(JSON, default=[])
    other_tech_stack = Column(JSON, default=[])

def save_and_update_target_accounts(accounts, product_id):
    
    db: Session = next(get_db())

    for account in accounts:
        print("Processing account:", account)
        account_name = account.get("account_name")
        industry = account.get("industry")
        revenue_range = account.get("revenue_range") or account.get("revenue")
        employee_range = account.get("employee_range") or account.get("employees")
        funding_stage = account.get("funding_stage") or account.get("funding")
        geography = account.get("geography") 
        competitor_used = account.get("competitor_used", [])
        other_tech_stack = account.get("other_tech_stack", [])


        if not account_name:
            print(f"Skipping account with missing name: {account}")
            continue

        # Check if the target account already exists for the given product_id
        existing_account = db.query(TargetAccount).filter_by(
            product_id=product_id,
            account_name=account_name,
        ).first()

        if existing_account:
            # Update existing record
            existing_account.industry = industry
            existing_account.revenue_range = revenue_range
            existing_account.employee_range = employee_range
            existing_account.funding_stage = funding_stage
            existing_account.geography = geography
            existing_account.competitor_used = competitor_used
            existing_account.other_tech_stack = other_tech_stack
            print(f"Updated existing account: {account_name}")
        else:
            # Create new record
            new_account = TargetAccount(
                id=str(uuid.uuid4()),
                product_id=product_id,
                account_name=account_name,
                industry=industry,
                revenue_range=revenue_range,
                employee_range=employee_range,
                funding_stage=funding_stage,
                geography=geography,
                competitor_used=competitor_used,
                other_tech_stack=other_tech_stack
            )
            db.add(new_account)
            print(f"Added new account: {account_name}")

    db.commit()
    db.close()

def load_target_accounts_from_db(product_id):
    db: Session = next(get_db())
    accounts = db.query(TargetAccount).filter_by(product_id=product_id).all()
    db.close()
    # Serialize for API response
    return [account_to_dict(a) for a in accounts]

def account_to_dict(account):
    return {
        "id": account.id,
        "product_id": account.product_id,
        "account_name": account.account_name,
        "industry": account.industry,
        "revenue_range": account.revenue_range,
        "employee_range": account.employee_range,
        "funding_stage": account.funding_stage,
        "geography": account.geography,
        "competitor_used": account.competitor_used or [],
        "other_tech_stack": account.other_tech_stack or []
    }
def delete_target_account_handler(account_id, product_id):
    print("Deletion loop inside handler")
    db: Session = next(get_db())
    all_accounts = db.query(TargetAccount).filter_by(product_id=product_id).all()
    print("Available account IDs for product:", [a.id for a in all_accounts])
    account = db.query(TargetAccount).filter_by(id=account_id, product_id=product_id).first()
    if not account:
        db.close()
        print("No account found to delete")
        return False
    db.delete(account)
    db.commit()
    db.close()
    return True