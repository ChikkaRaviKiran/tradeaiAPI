import base64, json
from app.core.config import settings
t = settings.dhan_access_token or ""
parts = t.split(".")
if len(parts) >= 2:
    p = parts[1] + "=" * (-len(parts[1]) % 4)
    payload = json.loads(base64.urlsafe_b64decode(p))
    print("in-memory token payload:", payload)
print("settings.dhan_client_id:", settings.dhan_client_id)
