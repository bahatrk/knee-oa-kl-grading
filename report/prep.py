import pandas as pd, numpy as np, json
U="/mnt/user-data/uploads/"; X="/home/claude/xai_v2/"
m=pd.read_csv(U+"9025e821-main_table_test_raw.csv").set_index("run")
lat=pd.read_csv(X+"cpu_latency.csv").set_index("run")
res=pd.read_csv(U+"f3c765f7-results.csv")
bt=pd.read_csv(X+"baseline_test.csv").iloc[0]
order=[("SimpleCNN","SimpleCNN","CNN, scratch"),("MiniResNet","MiniResNet","CNN, scratch (residual)"),("ResNet50__opt","ResNet50*","CNN, transfer"),
("DenseNet121","DenseNet121","CNN, transfer"),("EfficientNetB0","EfficientNet-B0","CNN, transfer"),("MobileNetV3L","MobileNetV3-L","CNN, transfer"),
("Xception","Xception","CNN, transfer"),("InceptionV3","Inception-v3","CNN, transfer"),("ConvNeXtTiny__opt","ConvNeXt-Tiny*","CNN, transfer (modern)"),
("DeiTSmall__opt","DeiT-Small*","Transformer, transfer"),("Orig_base__opt","Control: plain ConvNeXt-T*","CNN, transfer (hp of Original)"),
("Orig_full__opt","KneeOrdinalNet (Original)*","CNN + custom components"),("hybrid_mean4","Hybrid (mean of 4)*","Ensemble")]
mets=["accuracy","balanced_accuracy","macro_precision","macro_recall","macro_f1","cohen_kappa","mcc","roc_auc_macro"]
hog=[bt.accuracy,bt.balanced_accuracy,bt.macro_precision,bt.macro_recall,bt.macro_f1,bt.cohen_kappa,bt.mcc,bt.roc_auc_macro]
raw=[dict(name="HOG+Linear SVM",fam="Classical",params=6485,size=None,lat=None,v=[(x,None) for x in hog])]
for run,nm,fam in order:
    r=m.loc[run]; v=[(r[k+"_mean"],r[k+"_std"] if r["n_seeds"]>1 else None) for k in mets]
    p=27823973 if run=="Orig_base__opt" else (int(lat.loc[run,"params"]))
    L=lat.loc[run] if run in lat.index else None
    raw.append(dict(name=nm,fam=fam,params=p,size=None if L is None else L.size_mb,lat=None if L is None else L.latency_ms_median,p95=None if L is None else L.latency_ms_p95,v=v))
nn=raw[1:]
best=[max(x["v"][i][0] for x in raw) for i in range(8)]
bp=min(x["params"] for x in nn); bs=min(x["size"] for x in nn if x["size"]); bl=min(x["lat"] for x in nn if x["lat"])
main=[]
for x in raw:
    h=x["name"].startswith("HOG")
    row=[x["name"],x["fam"],"224²×%d"%(1 if h else 3),("**%.2f**" if x["params"]==bp and not h else "%.2f")%(x["params"]/1e6)]
    for i,(a,s) in enumerate(x["v"]):
        t="%.3f"%a+("" if s is None else "±%.3f"%s)
        row.append("**"+t+"**" if abs(a-best[i])<1e-9 else t)
    row.append("–" if x["size"] is None else (("**%.1f**" if x["size"]==bs else "%.1f")%x["size"]))
    row.append("–" if x["lat"] is None else (("**%.0f / %.0f**" if x["lat"]==bl else "%.0f / %.0f")%(x["lat"],x["p95"])))
    main.append(row)
ab=pd.read_csv(U+"628af9f6-ablation_summary.csv").set_index("run")
def f(run,k): return "%.3f±%.3f"%(m.loc[run,k+"_mean"],m.loc[run,k+"_std"])
abl=[]
for run,nm in [("Orig_full__opt","Full model"),("Orig_noMS__opt","− multi-scale pooling (B)"),("Orig_noRow__opt","− row attention (C)"),("Orig_noOrd__opt","− ordinal loss (D → CE)"),("Orig_v3__opt","− per-image standardization (A)"),("Orig_base__opt","Control (none of A–D)")]:
    vf=ab.loc[run,"val_macro_f1_mean"] if run in ab.index else 0.6987
    pcn=27823973 if run=="Orig_base__opt" else int(res[res.model_name==run].param_count.iloc[0])
    abl.append([nm,"%.2f"%(pcn/1e6),"%.3f"%vf,f(run,"macro_f1"),f(run,"qwk"),f(run,"mae_grade"),f(run,"distant_error_rate"),f(run,"ece15")])
pc=pd.read_csv(X+"per_class_test.csv"); pcrows=[]
for run,nm in [("DeiTSmall__opt","DeiT-Small"),("Orig_base__opt","Control"),("Orig_full__opt","Original"),("hybrid_mean4","Hybrid")]:
    g=pc[pc.run==run].groupby("class")[["recall_sensitivity","f1"]].mean()
    pcrows.append([nm]+["%.2f / %.2f"%(g.loc[k,"recall_sensitivity"],g.loc[k,"f1"]) for k in ["KL0","KL1","KL2","KL3","KL4"]])
hp=[]
for run,nm in [("ResNet50__opt","ResNet50"),("ConvNeXtTiny__opt","ConvNeXt-Tiny"),("DeiTSmall__opt","DeiT-Small"),("Orig_full__opt","KneeOrdinalNet and control")]:
    r=res[(res.model_name==run)&(res.seed==42)].iloc[0]; h=json.loads(r.hyperparams)
    hp.append([nm,"%d"%r.batch_size,"%.1e / %.1e"%(h["lr_head"],h["lr_ft"]),"%.4f"%h["weight_decay"],"%.3f"%h["label_smoothing"],"%.2f"%h["drop"],"%.2f"%h["unfreeze_frac"],("%.3f / %.3f"%(h["ordinal_sigma"],h["ordinal_lambda"])) if run=="Orig_full__opt" else "–"])
def nz(x,fmt): return "–" if pd.isna(x) else fmt%x
tt=[]
for run,nm in [("HOG+LinearSVM","HOG+Linear SVM"),("SimpleCNN","SimpleCNN"),("MiniResNet","MiniResNet"),("ResNet50","ResNet50 (default)"),("DenseNet121","DenseNet121"),("EfficientNetB0","EfficientNet-B0"),("MobileNetV3L","MobileNetV3-L"),("Xception","Xception"),("InceptionV3","Inception-v3"),("ConvNeXtTiny","ConvNeXt-T (default)"),("DeiTSmall","DeiT-S (default)"),("ResNet50__opt","ResNet50 (tuned)"),("ConvNeXtTiny__opt","ConvNeXt-T (tuned)"),("DeiTSmall__opt","DeiT-S (tuned)"),("Orig_full__opt","KneeOrdinalNet (tuned)"),("Orig_base__opt","Control (tuned)")]:
    r=res[(res.model_name==run)&(res.seed==42)].iloc[0]; hh=json.loads(r.hyperparams)
    tt.append([nm,nz(r.epochs,"%d"),str(hh.get("best_epoch","–")),"%.1f"%(r.train_time_s/60),nz(r.val_macro_f1,"%.3f")])
# kalibrasyon (3 seed ortalaması) ve eşli bootstrap özeti
cb=pd.read_csv(X+"calibration.csv"); cal=[]
for run,nm in [("DeiTSmall__opt","DeiT-Small"),("Orig_base__opt","Control"),("Orig_full__opt","Original"),
               ("Orig_noOrd__opt","Original, CE loss (noOrd)"),("hybrid_mean4","Hybrid")]:
    g=cb[cb.run==run]
    cal.append([nm]+["%.3f"%g[c].mean() for c in ("ece15","ece15_temp","brier","brier_temp")]+["%.2f"%g["T"].mean()])
pb=pd.read_csv(X+"paired_bootstrap.csv"); pair=[]
for a,b,nm in [("Orig_full__opt","DeiTSmall__opt","Original vs DeiT-Small (best existing)"),("Orig_full__opt","ConvNeXtTiny__opt","Original vs ConvNeXt-Tiny"),
               ("Orig_full__opt","Orig_base__opt","Original vs control (same hp)"),("hybrid_mean4","DeiTSmall__opt","Hybrid vs DeiT-Small"),
               ("hybrid_mean4","Orig_full__opt","Hybrid vs Original")]:
    row=[nm]
    for met in ("macro_f1","qwk","mae_grade"):
        g=pb[(pb.A==a)&(pb.B==b)&(pb.metric==met)]
        row.append("%+.3f (%d/%d)"%(g.diff_A_minus_B.mean(),int(g.ci_excludes_0.sum()),len(g)))
    pair.append(row)
json.dump(dict(main=main,abl=abl,pc=pcrows,hp=hp,tt=tt,cal=cal,pair=pair,total_h=float(res.train_time_s.sum()/3600)),open("data.json","w"),indent=1,ensure_ascii=False)
print(res.train_time_s.sum()/3600)
for r in main: print(r)
for r in abl: print(r)
for r in pcrows: print(r)
for r in hp: print(r)
