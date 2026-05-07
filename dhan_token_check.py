from app.core.config import settings
t = settings.dhan_access_token or ''
print('settings.dhan_access_token tail:', t[-12:] if t else 'EMPTY', 'len=', len(t))
print('settings.dhan_client_id:', settings.dhan_client_id)
