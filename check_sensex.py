import sys; sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')
from app.data.angelone_client import AngelOneClient

a = AngelOneClient()
a.authenticate()
instruments = a._get_instrument_master()

# Find SENSEX spot token on BSE
print("=== BSE SENSEX Spot ===")
for i in instruments:
    if 'SENSEX' in i.get('name', '') and i.get('exch_seg') == 'BSE':
        print(f"  {i}")
        break

# Find SENSEX on BFO
print("\n=== BFO SENSEX Options ===")
bfo_sensex = [i for i in instruments if i.get('exch_seg') == 'BFO' and 'SENSEX' in i.get('name', '')]
if bfo_sensex:
    print(f"  Total: {len(bfo_sensex)}")
    for x in bfo_sensex[:5]:
        print(f"  {x.get('symbol')} token={x.get('token')} expiry={x.get('expiry')} strike={x.get('strike')} lot={x.get('lotsize')}")
    expiries = sorted(set(x.get('expiry') for x in bfo_sensex))
    print(f"  Expiries: {expiries[:5]}")
    strikes = sorted(set(float(x.get('strike', 0)) / 100 for x in bfo_sensex if x.get('strike')))
    if len(strikes) > 2:
        diffs = [strikes[i+1] - strikes[i] for i in range(min(10, len(strikes) - 1))]
        print(f"  Sample strikes: {strikes[:10]}")
        print(f"  Strike diffs: {diffs}")
        print(f"  Strike interval: {min(d for d in diffs if d > 0)}")
    print(f"  Lot size: {bfo_sensex[0].get('lotsize')}")
else:
    print("  No BFO SENSEX found")

# Also check for SENSEX spot token on BSE index
print("\n=== Symbol Format Analysis ===")
# Separate weekly vs monthly
weekly = [i for i in bfo_sensex if i.get('lotsize') == '20']
monthly = [i for i in bfo_sensex if i.get('lotsize') != '20']
print(f"Weekly (lot=20): {len(weekly)}")
print(f"Monthly (lot!=20): {len(monthly)}")
for x in weekly[:5]:
    print(f"  W: {x.get('symbol')} expiry={x.get('expiry')}")
for x in monthly[:5]:
    print(f"  M: {x.get('symbol')} expiry={x.get('expiry')}")

# Expiry day of week
from datetime import datetime
for exp_str in sorted(set(x.get('expiry') for x in bfo_sensex))[:8]:
    try:
        d = datetime.strptime(exp_str, "%d%b%Y")
        print(f"  {exp_str} = {d.strftime('%A')}")
    except: pass
