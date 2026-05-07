from app.core.config import settings
print("KITE_PROXY_URL:", repr(getattr(settings, "kite_proxy_url", "<NOT SET>")))
print("KITE_API_KEY tail:", (settings.kite_api_key or "")[-6:])
print("KITE_ACCESS_TOKEN set:", bool(settings.kite_access_token))
