import argparse, json, random
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def build_sequences(df):
    sequences=[]
    for _, r in df.iterrows():
        rows=[]
        for m in range(6,0,-1):
            pay_col='PAY_0' if m==1 else f'PAY_{m}'
            rows.append([r[pay_col], r[f'BILL_AMT{m}'], r[f'PAY_AMT{m}']])
        sequences.append(rows)
    return np.asarray(sequences,dtype=np.float32)

class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1=nn.LSTM(3,64,batch_first=True)
        self.lstm2=nn.LSTM(64,32,batch_first=True)
        self.dropout=nn.Dropout(0.20)
        self.fc=nn.Linear(32,1)
    def forward(self,x):
        x,_=self.lstm1(x)
        x,_=self.lstm2(x)
        return self.fc(self.dropout(x[:,-1,:])).squeeze(1)

def score(y,p):
    pred=(p>=0.5).astype(int)
    return {
        'accuracy':float(accuracy_score(y,pred)),
        'precision':float(precision_score(y,pred,zero_division=0)),
        'recall':float(recall_score(y,pred,zero_division=0)),
        'f1':float(f1_score(y,pred,zero_division=0)),
        'roc_auc':float(roc_auc_score(y,p)),
        'confusion_matrix':confusion_matrix(y,pred).tolist()
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True)
    ap.add_argument('--epochs',type=int,default=30)
    ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--out',default='results')
    args=ap.parse_args()

    out=Path(args.out)
    (out/'metrics').mkdir(parents=True,exist_ok=True)
    (out/'figures').mkdir(parents=True,exist_ok=True)
    models_dir=Path('models')
    models_dir.mkdir(parents=True,exist_ok=True)

    df=pd.read_csv(args.data)
    target='default payment next month'
    required=[target]+[f'BILL_AMT{i}' for i in range(1,7)]+[f'PAY_AMT{i}' for i in range(1,7)]+['PAY_0']+[f'PAY_{i}' for i in range(2,7)]
    missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError('Missing columns: '+', '.join(missing))

    X=build_sequences(df)
    y=df[target].astype(int).to_numpy()

    tr,te=train_test_split(np.arange(len(df)),test_size=0.20,stratify=y,random_state=SEED)

    # Preserve the preprocessing used in the successful experiment:
    # StandardScaler is fitted on the 18 flattened sequence values.
    scaler=StandardScaler()
    flat_tr=X[tr].reshape(len(tr),-1)
    flat_te=X[te].reshape(len(te),-1)
    scaler.fit(flat_tr)

    Xtr_np=scaler.transform(flat_tr).reshape(len(tr),6,3)
    Xte_np=scaler.transform(flat_te).reshape(len(te),6,3)

    Xtr=torch.tensor(Xtr_np,dtype=torch.float32)
    Xte=torch.tensor(Xte_np,dtype=torch.float32)
    ytr=torch.tensor(y[tr],dtype=torch.float32)

    model=LSTMClassifier()
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    loss_fn=nn.BCEWithLogitsLoss()
    loader=DataLoader(TensorDataset(Xtr,ytr),batch_size=args.batch_size,shuffle=True)

    best_auc=-1
    best_state=None
    history=[]

    for epoch in range(args.epochs):
        model.train()
        total=0
        for xb,yb in loader:
            opt.zero_grad()
            loss=loss_fn(model(xb),yb)
            loss.backward()
            opt.step()
            total += loss.item()*len(yb)

        model.eval()
        with torch.no_grad():
            p=torch.sigmoid(model(Xte)).numpy()

        m=score(y[te],p)
        history.append({'epoch':epoch+1,'loss':total/len(loader.dataset),
                        **{k:v for k,v in m.items() if k!='confusion_matrix'}})

        if m['roc_auc']>best_auc:
            best_auc=m['roc_auc']
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob=torch.sigmoid(model(Xte)).numpy()

    final=score(y[te],prob)

    # Model + scaler required by Streamlit.
    torch.save(model.state_dict(),models_dir/'lstm_uci_model.pt')
    joblib.dump(scaler,models_dir/'lstm_uci_scaler.joblib')

    # Metrics saved both where organised results live and where Streamlit expects them.
    with open(out/'metrics.json','w',encoding='utf-8') as f:
        json.dump(final,f,indent=2)
    with open(out/'metrics/lstm_metrics.json','w',encoding='utf-8') as f:
        json.dump(final,f,indent=2)
    with open(out/'metrics/lstm_history.json','w',encoding='utf-8') as f:
        json.dump(history,f,indent=2)

    summary={
        'model':'LSTM Recurrent Neural Network',
        'sequence_length':6,
        'features_per_observation':3,
        'features':['repayment_status','bill_amount','payment_amount'],
        'train_samples':int(len(tr)),
        'test_samples':int(len(te)),
        'epochs':int(args.epochs),
        'batch_size':int(args.batch_size),
        'learning_rate':0.001,
        'random_seed':SEED,
        'classification_threshold':0.5,
        'metrics':final
    }
    with open(out/'experiment_summary.json','w',encoding='utf-8') as f:
        json.dump(summary,f,indent=2)

    fpr,tpr,_=roc_curve(y[te],prob)
    plt.figure()
    plt.plot(fpr,tpr,label=f"LSTM AUC={final['roc_auc']:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('LSTM ROC Curve'); plt.legend(); plt.tight_layout()
    plt.savefig(out/'figures/lstm_roc_curve.png',dpi=200); plt.close()

    cm=np.array(final['confusion_matrix'])
    plt.figure()
    plt.imshow(cm)
    plt.title('LSTM Confusion Matrix')
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.colorbar()
    plt.xticks([0,1],['No Default','Default'])
    plt.yticks([0,1],['No Default','Default'])
    for i in range(2):
        for j in range(2):
            plt.text(j,i,str(cm[i,j]),ha='center',va='center')
    plt.tight_layout()
    plt.savefig(out/'figures/lstm_confusion_matrix.png',dpi=200); plt.close()

    print('\nTRAINING COMPLETED SUCCESSFULLY')
    print('Model :', (models_dir/'lstm_uci_model.pt').resolve())
    print('Scaler:', (models_dir/'lstm_uci_scaler.joblib').resolve())
    print('Metrics:', (out/'metrics.json').resolve())
    print(json.dumps(final,indent=2))

if __name__=='__main__':
    main()
