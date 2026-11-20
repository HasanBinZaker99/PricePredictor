import math
from urllib.parse import urlsplit

from pydantic import  BaseModel, Field, field_validator

class Deal(BaseModel):
    product_description:str = Field(description="The product and its features")
    price: float = Field(description="The offered price in US dollars")
    url: str = Field(description="The products's deal link")

    # A decorator is a way to give a function extra behavior using an @ line above it. the main purpose of @field_validator is to tell Pydantic: "Use this function when checking Product_description"
    @field_validator("product_description")
    @classmethod
    def check_description(cls,value):
        # .strip() returns a string with whitespace removed from its beginning and end.
        if not value.strip():
            raise ValueError("A deal needs a product description.")
        return value.strip()

    @field_validator("price")
    @classmethod
    def check_price(cls,value):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("A deal price must be a finite number greater than zero.")
        return value

    @field_validator("url")
    @classmethod
    def check_url(cls,value):
        value = value.strip()
        parts = urlsplit(value)
        if parts.scheme not in ("http","https") or not parts.hostname or parts.username or parts.password:
            raise ValueError("A deal needs an ordinary http or https link.")
        return value