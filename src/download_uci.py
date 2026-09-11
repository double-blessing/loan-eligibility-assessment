from pathlib import Path
import urllib.request, zipfile
URL='https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip'
OUT=Path(__file__).resolve().parents[1]/'data/raw'; OUT.mkdir(parents=True,exist_ok=True)
zip_path=OUT/'uci_credit_card.zip'
urllib.request.urlretrieve(URL,zip_path)
with zipfile.ZipFile(zip_path) as z: z.extractall(OUT)
for p in OUT.iterdir():
    if p.suffix.lower()=='.xls':
        import pandas as pd
        pd.read_excel(p,header=1).to_csv(OUT/'UCI_Credit_Card.csv',index=False)
        print('Saved',OUT/'UCI_Credit_Card.csv'); break
