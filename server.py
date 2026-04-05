#!/usr/bin/env python3
"""
Shopify MCP Server — Full Admin API access via FastMCP.
Provides tools for managing products, orders, customers, collections,
inventory, and fulfillments through the Shopify Admin REST API.

Token Management:
  - Uses client_credentials grant to auto-generate and refresh tokens
  - Set SHOPIFY_CLIENT_ID + SHOPIFY_CLIENT_SECRET (recommended for OAuth apps)
  - Falls back to static SHOPIFY_ACCESS_TOKEN if client credentials not set
"""
import json
import os
import logging
import time
import asyncio
from typing import Optional, List, Dict, Any
from enum import Enum
import httpx
from pydantic import BaseModel, Field, ConfigDict, field_validator
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SHOPIFY_STORE        = os.environ.get("SHOPIFY_STORE", "")           # e.g. "my-store"
SHOPIFY_TOKEN        = os.environ.get("SHOPIFY_ACCESS_TOKEN", "")    # Static token (shpat_...)
SHOPIFY_CLIENT_ID    = os.environ.get("SHOPIFY_CLIENT_ID", "")
SHOPIFY_CLIENT_SECRET = os.environ.get("SHOPIFY_CLIENT_SECRET", "")
API_VERSION          = os.environ.get("SHOPIFY_API_VERSION", "2024-10")

# Refresh buffer: refresh token 30 minutes before expiry (only used with OAuth)
TOKEN_REFRESH_BUFFER = int(os.environ.get("TOKEN_REFRESH_BUFFER", "1800"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("shopify_mcp")

PORT          = int(os.environ.get("PORT", "8000"))
MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", "streamable-http")

mcp = FastMCP("shopify_mcp", host="0.0.0.0", port=PORT, json_response=True)


# ---------------------------------------------------------------------------
# Token Manager — handles automatic token lifecycle
# ---------------------------------------------------------------------------

class TokenManager:
    """
    Manages Shopify Admin API access tokens.

    Two modes:
      1. Static token  — set SHOPIFY_ACCESS_TOKEN (recommended for Custom Apps)
      2. OAuth / client_credentials — set SHOPIFY_CLIENT_ID + SHOPIFY_CLIENT_SECRET
         Enables auto-refresh before expiry and retry on 401.
    """

    def __init__(
        self,
        store: str,
        client_id: str,
        client_secret: str,
        static_token: str = "",
        refresh_buffer: int = 1800,
    ):
        self._store         = store
        self._client_id     = client_id
        self._client_secret = client_secret
        self._static_token  = static_token
        self._refresh_buffer = refresh_buffer

        self._access_token: str   = ""
        self._expires_at: float   = 0.0
        self._lock = asyncio.Lock()

        self._use_client_credentials = bool(client_id and client_secret)

        if self._use_client_credentials:
            logger.info("Token mode: client_credentials (auto-refresh enabled)")
        elif static_token:
            logger.info("Token mode: static SHOPIFY_ACCESS_TOKEN (no auto-refresh)")
            self._access_token = static_token
            self._expires_at   = float("inf")
        else:
            logger.warning(
                "No credentials configured. Set SHOPIFY_ACCESS_TOKEN or "
                "SHOPIFY_CLIENT_ID + SHOPIFY_CLIENT_SECRET."
            )

    @property
    def is_expired(self) -> bool:
        if not self._access_token:
            return True
        return time.time() >= (self._expires_at - self._refresh_buffer)

    async def get_token(self) -> str:
        if not self.is_expired:
            return self._access_token

        async with self._lock:
            if not self.is_expired:
                return self._access_token

            if self._use_client_credentials:
                await self._refresh_token()
            elif not self._access_token:
                raise RuntimeError(
                    "No valid token available. "
                    "Set SHOPIFY_ACCESS_TOKEN in your environment variables."
                )

        return self._access_token

    async def force_refresh(self) -> str:
        if not self._use_client_credentials:
            raise RuntimeError(
                "Cannot refresh — using a static token. "
                "Set SHOPIFY_CLIENT_ID + SHOPIFY_CLIENT_SECRET to enable auto-refresh."
            )
        async with self._lock:
            await self._refresh_token()
        return self._access_token

    async def _refresh_token(self) -> None:
        url = f"https://{self._store}.myshopify.com/admin/oauth/access_token"
        logger.info("Refreshing Shopify access token via client_credentials grant...")

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                data={
                    "grant_type":    "client_credentials",
                    "client_id":     self._client_id,
                    "client_secret": self._client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15.0,
            )

            if resp.status_code != 200:
                logger.error(f"Token refresh failed ({resp.status_code}): {resp.text[:500]}")
                raise RuntimeError(
                    f"Token refresh failed ({resp.status_code}). "
                    "Check SHOPIFY_CLIENT_ID and SHOPIFY_CLIENT_SECRET."
                )

            data               = resp.json()
            self._access_token = data["access_token"]
            expires_in         = data.get("expires_in", 86399)
            self._expires_at   = time.time() + expires_in

            scope         = data.get("scope", "")
            scope_preview = scope[:80] + "..." if len(scope) > 80 else scope
            logger.info(
                f"Token refreshed. Expires in {expires_in}s "
                f"({expires_in // 3600}h {(expires_in % 3600) // 60}m). "
                f"Scopes: {scope_preview}"
            )


# Global token manager
token_manager = TokenManager(
    store=SHOPIFY_STORE,
    client_id=SHOPIFY_CLIENT_ID,
    client_secret=SHOPIFY_CLIENT_SECRET,
    static_token=SHOPIFY_TOKEN,
    refresh_buffer=TOKEN_REFRESH_BUFFER,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _base_url() -> str:
    return f"https://{SHOPIFY_STORE}.myshopify.com/admin/api/{API_VERSION}"


async def _headers() -> dict:
    token = await token_manager.get_token()
    return {
        "X-Shopify-Access-Token": token,
        "Content-Type": "application/json",
    }


async def _request(
    method: str,
    path: str,
    params: Optional[dict] = None,
    body:   Optional[dict] = None,
    _retried: bool = False,
) -> dict:
    """Central HTTP helper — every API call flows through here.
    Auto-retries once on 401 when using OAuth credentials.
    """
    if not SHOPIFY_STORE:
        raise RuntimeError(
            "Missing SHOPIFY_STORE environment variable. "
            "Set it before starting the server."
        )

    url     = f"{_base_url()}/{path}"
    headers = await _headers()

    async with httpx.AsyncClient() as client:
        resp = await client.request(
            method, url,
            headers=headers,
            params=params,
            json=body,
            timeout=30.0,
        )

        if resp.status_code == 401 and not _retried and token_manager._use_client_credentials:
            logger.warning("Got 401 — refreshing token and retrying...")
            await token_manager.force_refresh()
            return await _request(method, path, params=params, body=body, _retried=True)

        resp.raise_for_status()
        if resp.status_code == 204:
            return {}
        return resp.json()


def _error(e: Exception) -> str:
    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text[:500]
        messages = {
            401: "Authentication failed — check your SHOPIFY_ACCESS_TOKEN (should start with shpat_).",
            403: "Permission denied — your token may be missing required API scopes.",
            404: "Resource not found — double-check the ID.",
            422: f"Validation error: {json.dumps(detail)}",
            429: "Rate-limited — wait a moment and retry.",
        }
        return messages.get(status, f"Shopify API error {status}: {json.dumps(detail)}")
    if isinstance(e, httpx.TimeoutException):
        return "Request timed out — try again."
    if isinstance(e, RuntimeError):
        return str(e)
    return f"Unexpected error: {type(e).__name__}: {e}"


def _fmt(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# PRODUCTS
# ═══════════════════════════════════════════════════════════════════════════

class ListProductsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    limit:          Optional[int]  = Field(default=50, ge=1, le=250, description="Max products to return (1-250)")
    status:         Optional[str]  = Field(default=None, description="Filter by status: active, archived, draft")
    product_type:   Optional[str]  = Field(default=None, description="Filter by product type")
    vendor:         Optional[str]  = Field(default=None, description="Filter by vendor name")
    collection_id:  Optional[int]  = Field(default=None, description="Filter by collection ID")
    since_id:       Optional[int]  = Field(default=None, description="Pagination: return products after this ID")
    fields:         Optional[str]  = Field(default=None, description="Comma-separated fields to include")


@mcp.tool(
    name="shopify_list_products",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_products(params: ListProductsInput) -> str:
    """List products from the Shopify store with optional filters."""
    try:
        p: Dict[str, Any] = {"limit": params.limit}
        for field in ["status", "product_type", "vendor", "collection_id", "since_id", "fields"]:
            val = getattr(params, field)
            if val is not None:
                p[field] = val
        data     = await _request("GET", "products.json", params=p)
        products = data.get("products", [])
        return _fmt({"count": len(products), "products": products})
    except Exception as e:
        return _error(e)


class GetProductInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: int = Field(..., description="The Shopify product ID")


@mcp.tool(
    name="shopify_get_product",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_product(params: GetProductInput) -> str:
    """Retrieve a single product by ID, including all variants and images."""
    try:
        data = await _request("GET", f"products/{params.product_id}.json")
        return _fmt(data.get("product", data))
    except Exception as e:
        return _error(e)


class CreateProductInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    title:        str                        = Field(..., min_length=1, description="Product title")
    body_html:    Optional[str]              = Field(default=None, description="HTML description")
    vendor:       Optional[str]              = Field(default=None)
    product_type: Optional[str]              = Field(default=None)
    tags:         Optional[str]              = Field(default=None, description="Comma-separated tags")
    status:       Optional[str]              = Field(default="draft", description="active, archived, or draft")
    variants:     Optional[List[Dict[str, Any]]] = Field(default=None, description="Variant objects with price, sku, etc.")
    options:      Optional[List[Dict[str, Any]]] = Field(default=None, description="Product options (Size, Color, etc.)")
    images:       Optional[List[Dict[str, Any]]] = Field(default=None, description="Image objects with src URL")


@mcp.tool(
    name="shopify_create_product",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_product(params: CreateProductInput) -> str:
    """Create a new product in the Shopify store."""
    try:
        product: Dict[str, Any] = {"title": params.title}
        for field in ["body_html", "vendor", "product_type", "tags", "status", "variants", "options", "images"]:
            val = getattr(params, field)
            if val is not None:
                product[field] = val
        data = await _request("POST", "products.json", body={"product": product})
        return _fmt(data.get("product", data))
    except Exception as e:
        return _error(e)


class UpdateProductInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    product_id:   int            = Field(..., description="Product ID to update")
    title:        Optional[str]  = Field(default=None)
    body_html:    Optional[str]  = Field(default=None)
    vendor:       Optional[str]  = Field(default=None)
    product_type: Optional[str]  = Field(default=None)
    tags:         Optional[str]  = Field(default=None)
    status:       Optional[str]  = Field(default=None, description="active, archived, or draft")
    variants:     Optional[List[Dict[str, Any]]] = Field(default=None)


@mcp.tool(
    name="shopify_update_product",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_update_product(params: UpdateProductInput) -> str:
    """Update an existing product. Only provided fields are changed."""
    try:
        product: Dict[str, Any] = {}
        for field in ["title", "body_html", "vendor", "product_type", "tags", "status", "variants"]:
            val = getattr(params, field)
            if val is not None:
                product[field] = val
        data = await _request("PUT", f"products/{params.product_id}.json", body={"product": product})
        return _fmt(data.get("product", data))
    except Exception as e:
        return _error(e)


class DeleteProductInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: int = Field(..., description="Product ID to delete")


@mcp.tool(
    name="shopify_delete_product",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_delete_product(params: DeleteProductInput) -> str:
    """Permanently delete a product. This cannot be undone."""
    try:
        await _request("DELETE", f"products/{params.product_id}.json")
        return f"Product {params.product_id} deleted."
    except Exception as e:
        return _error(e)


class ProductCountInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status:       Optional[str] = Field(default=None, description="active, archived, or draft")
    vendor:       Optional[str] = Field(default=None)
    product_type: Optional[str] = Field(default=None)


@mcp.tool(
    name="shopify_count_products",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_count_products(params: ProductCountInput) -> str:
    """Get the total count of products, optionally filtered."""
    try:
        p: Dict[str, Any] = {}
        for field in ["status", "vendor", "product_type"]:
            val = getattr(params, field)
            if val is not None:
                p[field] = val
        data = await _request("GET", "products/count.json", params=p)
        return _fmt(data)
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# ORDERS
# ═══════════════════════════════════════════════════════════════════════════

class ListOrdersInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    limit:               Optional[int] = Field(default=50, ge=1, le=250)
    status:              Optional[str] = Field(default="any", description="open, closed, cancelled, any")
    financial_status:    Optional[str] = Field(default=None, description="authorized, pending, paid, refunded, voided, any")
    fulfillment_status:  Optional[str] = Field(default=None, description="shipped, partial, unshipped, unfulfilled, any")
    since_id:            Optional[int] = Field(default=None)
    created_at_min:      Optional[str] = Field(default=None, description="ISO 8601 date, e.g. 2024-01-01T00:00:00Z")
    created_at_max:      Optional[str] = Field(default=None)
    fields:              Optional[str] = Field(default=None)


@mcp.tool(
    name="shopify_list_orders",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_orders(params: ListOrdersInput) -> str:
    """List orders with optional filters for status, financial/fulfillment status, and date range."""
    try:
        p: Dict[str, Any] = {"limit": params.limit, "status": params.status}
        for field in ["financial_status", "fulfillment_status", "since_id", "created_at_min", "created_at_max", "fields"]:
            val = getattr(params, field)
            if val is not None:
                p[field] = val
        data   = await _request("GET", "orders.json", params=p)
        orders = data.get("orders", [])
        return _fmt({"count": len(orders), "orders": orders})
    except Exception as e:
        return _error(e)


class GetOrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: int = Field(..., description="The Shopify order ID")


@mcp.tool(
    name="shopify_get_order",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_order(params: GetOrderInput) -> str:
    """Retrieve a single order by ID with full details."""
    try:
        data = await _request("GET", f"orders/{params.order_id}.json")
        return _fmt(data.get("order", data))
    except Exception as e:
        return _error(e)


class OrderCountInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status:             Optional[str] = Field(default="any")
    financial_status:   Optional[str] = Field(default=None)
    fulfillment_status: Optional[str] = Field(default=None)


@mcp.tool(
    name="shopify_count_orders",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_count_orders(params: OrderCountInput) -> str:
    """Get total order count, optionally filtered."""
    try:
        p: Dict[str, Any] = {"status": params.status}
        for field in ["financial_status", "fulfillment_status"]:
            val = getattr(params, field)
            if val is not None:
                p[field] = val
        data = await _request("GET", "orders/count.json", params=p)
        return _fmt(data)
    except Exception as e:
        return _error(e)


class CloseOrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: int = Field(..., description="Order ID to close")


@mcp.tool(
    name="shopify_close_order",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_close_order(params: CloseOrderInput) -> str:
    """Close an order (marks it as completed)."""
    try:
        data = await _request("POST", f"orders/{params.order_id}/close.json")
        return _fmt(data.get("order", data))
    except Exception as e:
        return _error(e)


class CancelOrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: int            = Field(..., description="Order ID to cancel")
    reason:   Optional[str]  = Field(default=None, description="customer, fraud, inventory, declined, other")
    email:    Optional[bool] = Field(default=True,  description="Send cancellation email to customer")
    restock:  Optional[bool] = Field(default=False, description="Restock line items")


@mcp.tool(
    name="shopify_cancel_order",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_cancel_order(params: CancelOrderInput) -> str:
    """Cancel an order. Optionally restock items and notify the customer."""
    try:
        body: Dict[str, Any] = {}
        for field in ["reason", "email", "restock"]:
            val = getattr(params, field)
            if val is not None:
                body[field] = val
        data = await _request("POST", f"orders/{params.order_id}/cancel.json", body=body)
        return _fmt(data.get("order", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# CUSTOMERS
# ═══════════════════════════════════════════════════════════════════════════

class ListCustomersInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    limit:          Optional[int] = Field(default=50, ge=1, le=250)
    since_id:       Optional[int] = Field(default=None)
    created_at_min: Optional[str] = Field(default=None, description="ISO 8601 date")
    created_at_max: Optional[str] = Field(default=None)
    fields:         Optional[str] = Field(default=None)


@mcp.tool(
    name="shopify_list_customers",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_customers(params: ListCustomersInput) -> str:
    """List customers from the store."""
    try:
        p: Dict[str, Any] = {"limit": params.limit}
        for f in ["since_id", "created_at_min", "created_at_max", "fields"]:
            val = getattr(params, f)
            if val is not None:
                p[f] = val
        data      = await _request("GET", "customers.json", params=p)
        customers = data.get("customers", [])
        return _fmt({"count": len(customers), "customers": customers})
    except Exception as e:
        return _error(e)


class SearchCustomersInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str           = Field(..., min_length=1, description="Search query (name, email, etc.)")
    limit: Optional[int] = Field(default=50, ge=1, le=250)


@mcp.tool(
    name="shopify_search_customers",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_search_customers(params: SearchCustomersInput) -> str:
    """Search customers by name, email, or other fields."""
    try:
        p         = {"query": params.query, "limit": params.limit}
        data      = await _request("GET", "customers/search.json", params=p)
        customers = data.get("customers", [])
        return _fmt({"count": len(customers), "customers": customers})
    except Exception as e:
        return _error(e)


class GetCustomerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: int = Field(..., description="Shopify customer ID")


@mcp.tool(
    name="shopify_get_customer",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_customer(params: GetCustomerInput) -> str:
    """Retrieve a single customer by ID."""
    try:
        data = await _request("GET", f"customers/{params.customer_id}.json")
        return _fmt(data.get("customer", data))
    except Exception as e:
        return _error(e)


class CreateCustomerInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    first_name:         Optional[str]  = Field(default=None)
    last_name:          Optional[str]  = Field(default=None)
    email:              Optional[str]  = Field(default=None)
    phone:              Optional[str]  = Field(default=None)
    tags:               Optional[str]  = Field(default=None)
    note:               Optional[str]  = Field(default=None)
    addresses:          Optional[List[Dict[str, Any]]] = Field(default=None)
    send_email_invite:  Optional[bool] = Field(default=False)


@mcp.tool(
    name="shopify_create_customer",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_customer(params: CreateCustomerInput) -> str:
    """Create a new customer."""
    try:
        customer: Dict[str, Any] = {}
        for field in ["first_name", "last_name", "email", "phone", "tags", "note", "addresses", "send_email_invite"]:
            val = getattr(params, field)
            if val is not None:
                customer[field] = val
        data = await _request("POST", "customers.json", body={"customer": customer})
        return _fmt(data.get("customer", data))
    except Exception as e:
        return _error(e)


class UpdateCustomerInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    customer_id: int           = Field(..., description="Customer ID to update")
    first_name:  Optional[str] = Field(default=None)
    last_name:   Optional[str] = Field(default=None)
    email:       Optional[str] = Field(default=None)
    phone:       Optional[str] = Field(default=None)
    tags:        Optional[str] = Field(default=None)
    note:        Optional[str] = Field(default=None)


@mcp.tool(
    name="shopify_update_customer",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_update_customer(params: UpdateCustomerInput) -> str:
    """Update an existing customer. Only provided fields are changed."""
    try:
        customer: Dict[str, Any] = {}
        for field in ["first_name", "last_name", "email", "phone", "tags", "note"]:
            val = getattr(params, field)
            if val is not None:
                customer[field] = val
        data = await _request("PUT", f"customers/{params.customer_id}.json", body={"customer": customer})
        return _fmt(data.get("customer", data))
    except Exception as e:
        return _error(e)


class CustomerOrdersInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: int           = Field(..., description="Customer ID")
    limit:       Optional[int] = Field(default=50, ge=1, le=250)
    status:      Optional[str] = Field(default="any")


@mcp.tool(
    name="shopify_get_customer_orders",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_customer_orders(params: CustomerOrdersInput) -> str:
    """Get all orders for a specific customer."""
    try:
        p      = {"limit": params.limit, "status": params.status}
        data   = await _request("GET", f"customers/{params.customer_id}/orders.json", params=p)
        orders = data.get("orders", [])
        return _fmt({"count": len(orders), "orders": orders})
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# COLLECTIONS (Custom + Smart)
# ═══════════════════════════════════════════════════════════════════════════

class ListCollectionsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit:           Optional[int] = Field(default=50, ge=1, le=250)
    since_id:        Optional[int] = Field(default=None)
    collection_type: Optional[str] = Field(default="custom", description="'custom' or 'smart'")


@mcp.tool(
    name="shopify_list_collections",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_collections(params: ListCollectionsInput) -> str:
    """List custom or smart collections."""
    try:
        endpoint = "custom_collections.json" if params.collection_type == "custom" else "smart_collections.json"
        p: Dict[str, Any] = {"limit": params.limit}
        if params.since_id:
            p["since_id"] = params.since_id
        data = await _request("GET", endpoint, params=p)
        key  = "custom_collections" if params.collection_type == "custom" else "smart_collections"
        collections = data.get(key, [])
        return _fmt({"count": len(collections), "collections": collections})
    except Exception as e:
        return _error(e)


class GetCollectionProductsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection_id: int           = Field(..., description="Collection ID")
    limit:         Optional[int] = Field(default=50, ge=1, le=250)


@mcp.tool(
    name="shopify_get_collection_products",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_collection_products(params: GetCollectionProductsInput) -> str:
    """Get all products in a specific collection."""
    try:
        p        = {"limit": params.limit, "collection_id": params.collection_id}
        data     = await _request("GET", "products.json", params=p)
        products = data.get("products", [])
        return _fmt({"count": len(products), "products": products})
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# INVENTORY
# ═══════════════════════════════════════════════════════════════════════════

class ListInventoryLocationsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


@mcp.tool(
    name="shopify_list_locations",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_locations(params: ListInventoryLocationsInput) -> str:
    """List all inventory locations for the store."""
    try:
        data      = await _request("GET", "locations.json")
        locations = data.get("locations", [])
        return _fmt({"count": len(locations), "locations": locations})
    except Exception as e:
        return _error(e)


class GetInventoryLevelsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    location_id:         Optional[int] = Field(default=None, description="Filter by location ID")
    inventory_item_ids:  Optional[str] = Field(default=None, description="Comma-separated inventory item IDs")


@mcp.tool(
    name="shopify_get_inventory_levels",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_inventory_levels(params: GetInventoryLevelsInput) -> str:
    """Get inventory levels for specific locations or inventory items."""
    try:
        p: Dict[str, Any] = {}
        if params.location_id:
            p["location_ids"] = params.location_id
        if params.inventory_item_ids:
            p["inventory_item_ids"] = params.inventory_item_ids
        data   = await _request("GET", "inventory_levels.json", params=p)
        levels = data.get("inventory_levels", [])
        return _fmt({"count": len(levels), "inventory_levels": levels})
    except Exception as e:
        return _error(e)


class SetInventoryLevelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inventory_item_id: int = Field(..., description="Inventory item ID")
    location_id:       int = Field(..., description="Location ID")
    available:         int = Field(..., description="Available quantity to set")


@mcp.tool(
    name="shopify_set_inventory_level",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_set_inventory_level(params: SetInventoryLevelInput) -> str:
    """Set the available inventory for an item at a location."""
    try:
        body = {
            "inventory_item_id": params.inventory_item_id,
            "location_id":       params.location_id,
            "available":         params.available,
        }
        data = await _request("POST", "inventory_levels/set.json", body=body)
        return _fmt(data.get("inventory_level", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# FULFILLMENTS
# ═══════════════════════════════════════════════════════════════════════════

class ListFulfillmentsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: int           = Field(..., description="Order ID")
    limit:    Optional[int] = Field(default=50, ge=1, le=250)


@mcp.tool(
    name="shopify_list_fulfillments",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_fulfillments(params: ListFulfillmentsInput) -> str:
    """List fulfillments for a specific order."""
    try:
        p            = {"limit": params.limit}
        data         = await _request("GET", f"orders/{params.order_id}/fulfillments.json", params=p)
        fulfillments = data.get("fulfillments", [])
        return _fmt({"count": len(fulfillments), "fulfillments": fulfillments})
    except Exception as e:
        return _error(e)


class CreateFulfillmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id:         int                        = Field(..., description="Order ID to fulfill")
    location_id:      int                        = Field(..., description="Location ID fulfilling from")
    tracking_number:  Optional[str]              = Field(default=None)
    tracking_company: Optional[str]              = Field(default=None, description="e.g. UPS, FedEx, USPS")
    tracking_url:     Optional[str]              = Field(default=None)
    line_items:       Optional[List[Dict[str, Any]]] = Field(default=None, description="Specific line items (omit for all)")
    notify_customer:  Optional[bool]             = Field(default=True, description="Send shipping notification email")


@mcp.tool(
    name="shopify_create_fulfillment",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_fulfillment(params: CreateFulfillmentInput) -> str:
    """Create a fulfillment for an order (ship items)."""
    try:
        fulfillment: Dict[str, Any] = {"location_id": params.location_id}
        for field in ["tracking_number", "tracking_company", "tracking_url", "line_items", "notify_customer"]:
            val = getattr(params, field)
            if val is not None:
                fulfillment[field] = val
        data = await _request(
            "POST",
            f"orders/{params.order_id}/fulfillments.json",
            body={"fulfillment": fulfillment},
        )
        return _fmt(data.get("fulfillment", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# SHOP INFO
# ═══════════════════════════════════════════════════════════════════════════

class EmptyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


@mcp.tool(
    name="shopify_get_shop",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_shop(params: EmptyInput) -> str:
    """Get store information: name, domain, plan, currency, timezone, etc."""
    try:
        data = await _request("GET", "shop.json")
        return _fmt(data.get("shop", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# WEBHOOKS
# ═══════════════════════════════════════════════════════════════════════════

class ListWebhooksInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: Optional[int] = Field(default=50, ge=1, le=250)
    topic: Optional[str] = Field(default=None, description="Filter by topic, e.g. orders/create")


@mcp.tool(
    name="shopify_list_webhooks",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_webhooks(params: ListWebhooksInput) -> str:
    """List configured webhooks."""
    try:
        p: Dict[str, Any] = {"limit": params.limit}
        if params.topic:
            p["topic"] = params.topic
        data     = await _request("GET", "webhooks.json", params=p)
        webhooks = data.get("webhooks", [])
        return _fmt({"count": len(webhooks), "webhooks": webhooks})
    except Exception as e:
        return _error(e)


class CreateWebhookInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    topic:   str           = Field(..., description="Webhook topic, e.g. orders/create, products/update")
    address: str           = Field(..., description="URL to receive the webhook POST")
    format:  Optional[str] = Field(default="json", description="json or xml")


@mcp.tool(
    name="shopify_create_webhook",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_webhook(params: CreateWebhookInput) -> str:
    """Create a new webhook subscription."""
    try:
        webhook = {"topic": params.topic, "address": params.address, "format": params.format}
        data    = await _request("POST", "webhooks.json", body={"webhook": webhook})
        return _fmt(data.get("webhook", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# THEMES & ASSETS (edit Liquid, CSS, JS, JSON)
# ═══════════════════════════════════════════════════════════════════════════

class ListThemesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


@mcp.tool(
    name="shopify_list_themes",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_themes(params: ListThemesInput) -> str:
    """List all themes. The one with role='main' is the live/published theme."""
    try:
        data = await _request("GET", "themes.json")
        themes = data.get("themes", [])
        return _fmt({"count": len(themes), "themes": themes})
    except Exception as e:
        return _error(e)


class ListThemeAssetsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    theme_id: int = Field(..., description="Theme ID")


@mcp.tool(
    name="shopify_list_theme_assets",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_theme_assets(params: ListThemeAssetsInput) -> str:
    """List all asset keys (files) in a theme. Returns file paths like 'layout/theme.liquid'."""
    try:
        data = await _request("GET", f"themes/{params.theme_id}/assets.json")
        assets = data.get("assets", [])
        return _fmt({"count": len(assets), "assets": assets})
    except Exception as e:
        return _error(e)


class GetThemeAssetInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    theme_id: int = Field(..., description="Theme ID")
    asset_key: str = Field(..., description="Asset key path, e.g. 'layout/theme.liquid', 'sections/header.liquid'")


@mcp.tool(
    name="shopify_get_theme_asset",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_theme_asset(params: GetThemeAssetInput) -> str:
    """Read the content of a theme file (Liquid, CSS, JS, JSON). Returns the full source code."""
    try:
        data = await _request("GET", f"themes/{params.theme_id}/assets.json", params={"asset[key]": params.asset_key})
        return _fmt(data.get("asset", data))
    except Exception as e:
        return _error(e)


class PutThemeAssetInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    theme_id: int = Field(..., description="Theme ID")
    asset_key: str = Field(..., description="Asset key path, e.g. 'sections/custom.liquid'")
    value: str = Field(..., description="Full file content (Liquid/HTML/CSS/JS/JSON)")


@mcp.tool(
    name="shopify_put_theme_asset",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_put_theme_asset(params: PutThemeAssetInput) -> str:
    """Create or update a theme file. Use this to edit Liquid templates, CSS, JS, or JSON settings."""
    try:
        body = {"asset": {"key": params.asset_key, "value": params.value}}
        data = await _request("PUT", f"themes/{params.theme_id}/assets.json", body=body)
        return _fmt(data.get("asset", data))
    except Exception as e:
        return _error(e)


class DeleteThemeAssetInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    theme_id: int = Field(..., description="Theme ID")
    asset_key: str = Field(..., description="Asset key path to delete")


@mcp.tool(
    name="shopify_delete_theme_asset",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_delete_theme_asset(params: DeleteThemeAssetInput) -> str:
    """Delete a theme file. Cannot be undone."""
    try:
        await _request("DELETE", f"themes/{params.theme_id}/assets.json", params={"asset[key]": params.asset_key})
        return f"Asset '{params.asset_key}' deleted from theme {params.theme_id}."
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════════════

class ListPagesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: Optional[int] = Field(default=50, ge=1, le=250)
    since_id: Optional[int] = Field(default=None)
    published_status: Optional[str] = Field(default=None, description="published, unpublished, any")


@mcp.tool(
    name="shopify_list_pages",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_pages(params: ListPagesInput) -> str:
    """List pages in the store."""
    try:
        p: Dict[str, Any] = {"limit": params.limit}
        if params.since_id:
            p["since_id"] = params.since_id
        if params.published_status:
            p["published_status"] = params.published_status
        data = await _request("GET", "pages.json", params=p)
        pages = data.get("pages", [])
        return _fmt({"count": len(pages), "pages": pages})
    except Exception as e:
        return _error(e)


class GetPageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page_id: int = Field(..., description="Page ID")


@mcp.tool(
    name="shopify_get_page",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_page(params: GetPageInput) -> str:
    """Get a single page by ID with full HTML content."""
    try:
        data = await _request("GET", f"pages/{params.page_id}.json")
        return _fmt(data.get("page", data))
    except Exception as e:
        return _error(e)


class CreatePageInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    title: str = Field(..., min_length=1, description="Page title")
    body_html: Optional[str] = Field(default=None, description="HTML content of the page")
    handle: Optional[str] = Field(default=None, description="URL handle (slug)")
    published: Optional[bool] = Field(default=True, description="Whether the page is published")
    template_suffix: Optional[str] = Field(default=None, description="Theme template suffix")
    metafield: Optional[Dict[str, Any]] = Field(default=None)


@mcp.tool(
    name="shopify_create_page",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_page(params: CreatePageInput) -> str:
    """Create a new page in the store."""
    try:
        page: Dict[str, Any] = {"title": params.title}
        for field in ["body_html", "handle", "published", "template_suffix", "metafield"]:
            val = getattr(params, field)
            if val is not None:
                page[field] = val
        data = await _request("POST", "pages.json", body={"page": page})
        return _fmt(data.get("page", data))
    except Exception as e:
        return _error(e)


class UpdatePageInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    page_id: int = Field(..., description="Page ID to update")
    title: Optional[str] = Field(default=None)
    body_html: Optional[str] = Field(default=None)
    handle: Optional[str] = Field(default=None)
    published: Optional[bool] = Field(default=None)
    template_suffix: Optional[str] = Field(default=None)


@mcp.tool(
    name="shopify_update_page",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_update_page(params: UpdatePageInput) -> str:
    """Update an existing page."""
    try:
        page: Dict[str, Any] = {}
        for field in ["title", "body_html", "handle", "published", "template_suffix"]:
            val = getattr(params, field)
            if val is not None:
                page[field] = val
        data = await _request("PUT", f"pages/{params.page_id}.json", body={"page": page})
        return _fmt(data.get("page", data))
    except Exception as e:
        return _error(e)


class DeletePageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page_id: int = Field(..., description="Page ID to delete")


@mcp.tool(
    name="shopify_delete_page",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_delete_page(params: DeletePageInput) -> str:
    """Permanently delete a page."""
    try:
        await _request("DELETE", f"pages/{params.page_id}.json")
        return f"Page {params.page_id} deleted."
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# BLOGS & ARTICLES
# ═══════════════════════════════════════════════════════════════════════════

class ListBlogsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: Optional[int] = Field(default=50, ge=1, le=250)


@mcp.tool(
    name="shopify_list_blogs",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_blogs(params: ListBlogsInput) -> str:
    """List all blogs in the store."""
    try:
        data = await _request("GET", "blogs.json", params={"limit": params.limit})
        blogs = data.get("blogs", [])
        return _fmt({"count": len(blogs), "blogs": blogs})
    except Exception as e:
        return _error(e)


class ListArticlesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    blog_id: int = Field(..., description="Blog ID")
    limit: Optional[int] = Field(default=50, ge=1, le=250)


@mcp.tool(
    name="shopify_list_articles",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_articles(params: ListArticlesInput) -> str:
    """List articles in a specific blog."""
    try:
        data = await _request("GET", f"blogs/{params.blog_id}/articles.json", params={"limit": params.limit})
        articles = data.get("articles", [])
        return _fmt({"count": len(articles), "articles": articles})
    except Exception as e:
        return _error(e)


class CreateArticleInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    blog_id: int = Field(..., description="Blog ID")
    title: str = Field(..., min_length=1)
    body_html: Optional[str] = Field(default=None)
    author: Optional[str] = Field(default=None)
    tags: Optional[str] = Field(default=None, description="Comma-separated tags")
    published: Optional[bool] = Field(default=True)
    image: Optional[Dict[str, Any]] = Field(default=None, description="Image object with src URL")


@mcp.tool(
    name="shopify_create_article",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_article(params: CreateArticleInput) -> str:
    """Create a new article in a blog."""
    try:
        article: Dict[str, Any] = {"title": params.title}
        for field in ["body_html", "author", "tags", "published", "image"]:
            val = getattr(params, field)
            if val is not None:
                article[field] = val
        data = await _request("POST", f"blogs/{params.blog_id}/articles.json", body={"article": article})
        return _fmt(data.get("article", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# DISCOUNT CODES & PRICE RULES
# ═══════════════════════════════════════════════════════════════════════════

class ListPriceRulesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: Optional[int] = Field(default=50, ge=1, le=250)


@mcp.tool(
    name="shopify_list_price_rules",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_price_rules(params: ListPriceRulesInput) -> str:
    """List all price rules (discount definitions)."""
    try:
        data = await _request("GET", "price_rules.json", params={"limit": params.limit})
        rules = data.get("price_rules", [])
        return _fmt({"count": len(rules), "price_rules": rules})
    except Exception as e:
        return _error(e)


class CreatePriceRuleInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    title: str = Field(..., description="Internal name for the price rule")
    target_type: str = Field(default="line_item", description="line_item or shipping_line")
    target_selection: str = Field(default="all", description="all or entitled")
    allocation_method: str = Field(default="across", description="across or each")
    value_type: str = Field(..., description="percentage or fixed_amount")
    value: str = Field(..., description="Negative number, e.g. '-10.0' for 10% off or -$10")
    customer_selection: str = Field(default="all", description="all or prerequisite")
    starts_at: Optional[str] = Field(default=None, description="ISO 8601 date")
    ends_at: Optional[str] = Field(default=None, description="ISO 8601 date")
    usage_limit: Optional[int] = Field(default=None)
    once_per_customer: Optional[bool] = Field(default=False)


@mcp.tool(
    name="shopify_create_price_rule",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_price_rule(params: CreatePriceRuleInput) -> str:
    """Create a price rule (discount definition). Then create a discount code for it."""
    try:
        rule: Dict[str, Any] = {}
        for field in ["title", "target_type", "target_selection", "allocation_method",
                       "value_type", "value", "customer_selection", "starts_at",
                       "ends_at", "usage_limit", "once_per_customer"]:
            val = getattr(params, field)
            if val is not None:
                rule[field] = val
        data = await _request("POST", "price_rules.json", body={"price_rule": rule})
        return _fmt(data.get("price_rule", data))
    except Exception as e:
        return _error(e)


class CreateDiscountCodeInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    price_rule_id: int = Field(..., description="Price rule ID to attach the code to")
    code: str = Field(..., description="The discount code customers will enter, e.g. 'SAVE10'")


@mcp.tool(
    name="shopify_create_discount_code",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_discount_code(params: CreateDiscountCodeInput) -> str:
    """Create a discount code for an existing price rule."""
    try:
        body = {"discount_code": {"code": params.code}}
        data = await _request("POST", f"price_rules/{params.price_rule_id}/discount_codes.json", body=body)
        return _fmt(data.get("discount_code", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# METAFIELDS
# ═══════════════════════════════════════════════════════════════════════════

class ListMetafieldsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resource: Optional[str] = Field(default=None, description="Resource type: products, collections, customers, orders, shop, pages, blogs, articles")
    resource_id: Optional[int] = Field(default=None, description="Resource ID (omit for shop-level)")
    namespace: Optional[str] = Field(default=None)
    limit: Optional[int] = Field(default=50, ge=1, le=250)


@mcp.tool(
    name="shopify_list_metafields",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_metafields(params: ListMetafieldsInput) -> str:
    """List metafields for a resource (product, collection, page, shop, etc.)."""
    try:
        p: Dict[str, Any] = {"limit": params.limit}
        if params.namespace:
            p["namespace"] = params.namespace
        if params.resource and params.resource_id:
            path = f"{params.resource}/{params.resource_id}/metafields.json"
        else:
            path = "metafields.json"
        data = await _request("GET", path, params=p)
        metafields = data.get("metafields", [])
        return _fmt({"count": len(metafields), "metafields": metafields})
    except Exception as e:
        return _error(e)


class CreateMetafieldInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    resource: Optional[str] = Field(default=None, description="Resource type")
    resource_id: Optional[int] = Field(default=None)
    namespace: str = Field(..., description="Metafield namespace, e.g. 'custom'")
    key: str = Field(..., description="Metafield key")
    value: str = Field(..., description="Metafield value")
    type: str = Field(default="single_line_text_field", description="Metafield type: single_line_text_field, number_integer, json, boolean, etc.")


@mcp.tool(
    name="shopify_create_metafield",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_metafield(params: CreateMetafieldInput) -> str:
    """Create a metafield on a resource or at shop level."""
    try:
        mf = {"namespace": params.namespace, "key": params.key, "value": params.value, "type": params.type}
        if params.resource and params.resource_id:
            path = f"{params.resource}/{params.resource_id}/metafields.json"
        else:
            path = "metafields.json"
        data = await _request("POST", path, body={"metafield": mf})
        return _fmt(data.get("metafield", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# REDIRECTS (URL redirects for SEO)
# ═══════════════════════════════════════════════════════════════════════════

class ListRedirectsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: Optional[int] = Field(default=50, ge=1, le=250)


@mcp.tool(
    name="shopify_list_redirects",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_redirects(params: ListRedirectsInput) -> str:
    """List URL redirects."""
    try:
        data = await _request("GET", "redirects.json", params={"limit": params.limit})
        redirects = data.get("redirects", [])
        return _fmt({"count": len(redirects), "redirects": redirects})
    except Exception as e:
        return _error(e)


class CreateRedirectInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    path: str = Field(..., description="Old URL path, e.g. '/old-page'")
    target: str = Field(..., description="New URL path or full URL, e.g. '/new-page'")


@mcp.tool(
    name="shopify_create_redirect",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_redirect(params: CreateRedirectInput) -> str:
    """Create a URL redirect (301)."""
    try:
        body = {"redirect": {"path": params.path, "target": params.target}}
        data = await _request("POST", "redirects.json", body=body)
        return _fmt(data.get("redirect", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# SCRIPT TAGS
# ═══════════════════════════════════════════════════════════════════════════

class ListScriptTagsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: Optional[int] = Field(default=50, ge=1, le=250)


@mcp.tool(
    name="shopify_list_script_tags",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_script_tags(params: ListScriptTagsInput) -> str:
    """List script tags injected into the storefront."""
    try:
        data = await _request("GET", "script_tags.json", params={"limit": params.limit})
        tags = data.get("script_tags", [])
        return _fmt({"count": len(tags), "script_tags": tags})
    except Exception as e:
        return _error(e)


class CreateScriptTagInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    src: str = Field(..., description="URL of the JavaScript file to inject")
    event: str = Field(default="onload", description="'onload' (default) — when to load the script")
    display_scope: Optional[str] = Field(default="all", description="'all', 'order_status', or 'online_store'")


@mcp.tool(
    name="shopify_create_script_tag",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_script_tag(params: CreateScriptTagInput) -> str:
    """Add a JavaScript tag to the storefront."""
    try:
        tag: Dict[str, Any] = {"src": params.src, "event": params.event}
        if params.display_scope:
            tag["display_scope"] = params.display_scope
        data = await _request("POST", "script_tags.json", body={"script_tag": tag})
        return _fmt(data.get("script_tag", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# COLLECTIONS (Create / Update)
# ═══════════════════════════════════════════════════════════════════════════

class CreateCustomCollectionInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    title: str = Field(..., min_length=1, description="Collection title")
    body_html: Optional[str] = Field(default=None, description="HTML description")
    published: Optional[bool] = Field(default=True)
    image: Optional[Dict[str, Any]] = Field(default=None, description="Image object with src URL")
    collects: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of {product_id: int} to add")


@mcp.tool(
    name="shopify_create_custom_collection",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_custom_collection(params: CreateCustomCollectionInput) -> str:
    """Create a new custom collection."""
    try:
        coll: Dict[str, Any] = {"title": params.title}
        for field in ["body_html", "published", "image", "collects"]:
            val = getattr(params, field)
            if val is not None:
                coll[field] = val
        data = await _request("POST", "custom_collections.json", body={"custom_collection": coll})
        return _fmt(data.get("custom_collection", data))
    except Exception as e:
        return _error(e)


class UpdateCustomCollectionInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    collection_id: int = Field(..., description="Collection ID")
    title: Optional[str] = Field(default=None)
    body_html: Optional[str] = Field(default=None)
    published: Optional[bool] = Field(default=None)


@mcp.tool(
    name="shopify_update_custom_collection",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_update_custom_collection(params: UpdateCustomCollectionInput) -> str:
    """Update an existing custom collection."""
    try:
        coll: Dict[str, Any] = {}
        for field in ["title", "body_html", "published"]:
            val = getattr(params, field)
            if val is not None:
                coll[field] = val
        data = await _request("PUT", f"custom_collections/{params.collection_id}.json", body={"custom_collection": coll})
        return _fmt(data.get("custom_collection", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# COLLECTS (add/remove products from collections)
# ═══════════════════════════════════════════════════════════════════════════

class AddProductToCollectionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: int = Field(..., description="Product ID to add")
    collection_id: int = Field(..., description="Collection ID to add the product to")


@mcp.tool(
    name="shopify_add_product_to_collection",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_add_product_to_collection(params: AddProductToCollectionInput) -> str:
    """Add a product to a custom collection."""
    try:
        body = {"collect": {"product_id": params.product_id, "collection_id": params.collection_id}}
        data = await _request("POST", "collects.json", body=body)
        return _fmt(data.get("collect", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# IMAGES (product images management)
# ═══════════════════════════════════════════════════════════════════════════

class ListProductImagesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: int = Field(..., description="Product ID")


@mcp.tool(
    name="shopify_list_product_images",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_product_images(params: ListProductImagesInput) -> str:
    """List all images for a product."""
    try:
        data = await _request("GET", f"products/{params.product_id}/images.json")
        images = data.get("images", [])
        return _fmt({"count": len(images), "images": images})
    except Exception as e:
        return _error(e)


class CreateProductImageInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    product_id: int = Field(..., description="Product ID")
    src: Optional[str] = Field(default=None, description="Image URL to upload from")
    attachment: Optional[str] = Field(default=None, description="Base64-encoded image data")
    filename: Optional[str] = Field(default=None, description="Filename for the image")
    alt: Optional[str] = Field(default=None, description="Alt text")
    position: Optional[int] = Field(default=None, description="Image position/order")
    variant_ids: Optional[List[int]] = Field(default=None, description="Variant IDs to associate")


@mcp.tool(
    name="shopify_create_product_image",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_product_image(params: CreateProductImageInput) -> str:
    """Add an image to a product. Provide either src (URL) or attachment (base64)."""
    try:
        image: Dict[str, Any] = {}
        for field in ["src", "attachment", "filename", "alt", "position", "variant_ids"]:
            val = getattr(params, field)
            if val is not None:
                image[field] = val
        data = await _request("POST", f"products/{params.product_id}/images.json", body={"image": image})
        return _fmt(data.get("image", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# SMART COLLECTIONS
# ═══════════════════════════════════════════════════════════════════════════

class CreateSmartCollectionInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    title: str = Field(..., min_length=1)
    rules: List[Dict[str, Any]] = Field(..., description="List of rules, e.g. [{'column': 'tag', 'relation': 'equals', 'condition': 'sale'}]")
    disjunctive: Optional[bool] = Field(default=False, description="True = match ANY rule, False = match ALL rules")
    published: Optional[bool] = Field(default=True)
    body_html: Optional[str] = Field(default=None)


@mcp.tool(
    name="shopify_create_smart_collection",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_smart_collection(params: CreateSmartCollectionInput) -> str:
    """Create an automated (smart) collection with rules."""
    try:
        coll: Dict[str, Any] = {"title": params.title, "rules": params.rules}
        for field in ["disjunctive", "published", "body_html"]:
            val = getattr(params, field)
            if val is not None:
                coll[field] = val
        data = await _request("POST", "smart_collections.json", body={"smart_collection": coll})
        return _fmt(data.get("smart_collection", data))
    except Exception as e:
        return _error(e)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run(transport=MCP_TRANSPORT)
