import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, HTMLResponse

app = FastAPI(title="Next Level Decor Product Redirect")

SHOPIFY_BASE = "https://nextleveldecor.in/products/"
REDIRECTS_PATH = Path(__file__).parent / "redirects.json"

_redirects: dict = {}


@app.on_event("startup")
def load_redirects():
    global _redirects
    if REDIRECTS_PATH.exists():
        with open(REDIRECTS_PATH) as f:
            data = json.load(f)
        _redirects = {k.upper(): v for k, v in data.items()}
    print(f"Loaded {len(_redirects)} product redirects")


@app.get("/health")
def health():
    return {"status": "ok", "products": len(_redirects)}


@app.get("/{sku}")
def redirect_sku(sku: str):
    handle = _redirects.get(sku.upper())
    if handle:
        return RedirectResponse(url=f"{SHOPIFY_BASE}{handle}", status_code=302)
    return HTMLResponse(
        content=f"""<html><body style="font-family:sans-serif;text-align:center;padding:60px">
<h2>Product Coming Soon</h2>
<p>SKU: <strong>{sku}</strong></p>
<p>This product will be available shortly on our store.</p>
<br>
<a href="https://nextleveldecor.in" style="font-size:16px">Browse Next Level Decor &rarr;</a>
</body></html>""",
        status_code=404,
    )
