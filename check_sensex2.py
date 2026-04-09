import sys; sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')
from app.data.angelone_client import AngelOneClient

a = AngelOneClient()
a.authenticate()
instruments = a._get_instrument_master()
bfo = [i for i in instruments if i.get('exch_seg') == 'BFO' and 'SENSEX' in i.get('name', '') and i.get('lotsize') == '20']

# 09APR2026 expiry
apr9 = [i for i in bfo if i.get('expiry') == '09APR2026']
print(f"09APR2026 weekly options: {len(apr9)}")
for x in sorted(apr9, key=lambda x: float(x.get('strike', 0)))[:5]:
    s = float(x.get('strike', 0)) / 100
    print(f"  {x['symbol']}  strike={s:.0f}")

# 16APR2026 expiry
apr16 = [i for i in bfo if i.get('expiry') == '16APR2026']
print(f"\n16APR2026 weekly options: {len(apr16)}")
for x in sorted(apr16, key=lambda x: float(x.get('strike', 0)))[:5]:
    s = float(x.get('strike', 0)) / 100
    print(f"  {x['symbol']}  strike={s:.0f}")

# Near ATM strike interval
near_atm = [i for i in apr9 if 7900000 < float(i.get('strike', 0)) < 8100000]
strikes_near = sorted(set(float(x.get('strike', 0)) / 100 for x in near_atm))
print(f"\nNear ATM strikes (79000-81000): {strikes_near[:15]}")
if len(strikes_near) > 1:
    print(f"Interval: {strikes_near[1] - strikes_near[0]}")
