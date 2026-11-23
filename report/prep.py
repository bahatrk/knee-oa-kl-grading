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

# ---------------- yeni tablolar (proje belgesi: Tablo 2 biçimi, klinik bakış, güven aralıkları, ek tablolar) ----------------
R=pd.read_csv("/home/claude/results_new.csv"); L=pd.read_csv(X+"cpu_latency.csv").set_index("run"); LB=pd.read_csv(X+"cpu_latency_extra.csv").set_index("run")
def ms(run,col,fmt="%.3f"):
    g=R[R.model_name==run][col].astype(float); return (fmt%g.mean())+"±"+(fmt%g.std(ddof=1))
def dlt(d): return "0.000" if abs(d)<0.0005 else ("%+.3f"%d).replace("-","−")
def mn(run,col): return R[R.model_name==run][col].astype(float).mean()
full_val=mn("Orig_full__opt","val_macro_f1")
abl2=[]
for exp,run,nm in [("Full model","Orig_full__opt","KneeOrdinalNet (A–D)"),("Ablation 1","Orig_noMS__opt","− multi-scale pooling (B)"),
                   ("Ablation 2","Orig_noRow__opt","− row attention (C)"),("Ablation 3","Orig_noOrd__opt","− ordinal loss (D → CE)"),
                   ("Ablation 4","Orig_v3__opt","− per-image standardization (A)"),("Control","Orig_base__opt","none of A–D (plain ConvNeXt-T)")]:
    d=mn(run,"val_macro_f1")-full_val
    abl2.append([exp,nm,"%.2f"%(mn(run,"param_count")/1e6),ms(run,"val_macro_f1"),ms(run,"val_balanced_accuracy"),(dlt(d) if exp!="Full model" else "–"),
                 ms(run,"test_macro_f1"),ms(run,"test_balanced_accuracy"),"%.1f"%LB.loc[run,"latency_ms_median"]])
# hibrit ablasyonu: üyeler tek başına ve füzyon yöntemleri (doğrulama; gecikme ilk oturumda, hibrit = üyelerin toplamı)
hs=pd.read_csv(U+"5f90692c-hybrid_val_summary.csv"); h4=hs[hs.n_members==4].set_index("method"); mean_val=h4.loc["mean","macro_f1_mean"]
members=[("Orig_full__opt","KneeOrdinalNet alone"),("DeiTSmall__opt","DeiT-Small alone"),("ConvNeXtTiny__opt","ConvNeXt-Tiny alone"),("ResNet50__opt","ResNet50 alone")]
habl=[]
for run,nm in members:
    habl.append([nm,"%.2f"%(mn(run,"param_count")/1e6),ms(run,"val_macro_f1"),ms(run,"val_balanced_accuracy"),dlt(mn(run,"val_macro_f1")-mean_val),"%.1f"%L.loc[run,"latency_ms_median"]])
hp4=sum(mn(r,"param_count") for r,_ in members)/1e6; hl4=L.loc["hybrid_mean4","latency_ms_median"]
for meth,nm in [("mean","4 members, equal-weight mean (selected)"),("weighted","4 members, weighted mean"),("stack","4 members, stacking (logistic)"),("mean_ord","4 members, mean + ordinal decision rule")]:
    r=h4.loc[meth]
    habl.append([nm,"%.2f"%hp4,"%.3f±%.3f"%(r.macro_f1_mean,r.macro_f1_std),"%.3f±%.3f"%(r.balanced_accuracy_mean,r.balanced_accuracy_std),(dlt(r.macro_f1_mean-mean_val) if meth!="mean" else "–"),"%.1f"%hl4])
# klinik bakış: KL>=2 (kesin radyografik OA) eşiği, karışıklık matrislerinden; 3 seed ortalama±SD
def conf(run,seed): return np.load(X+"confusion_%s_seed%d.npy"%(run,seed))
def binstats(run,seeds):
    out=[]
    for sd in seeds:
        c=conf(run,sd); pos=c[2:].sum(); neg=c[:2].sum()
        tot=c.sum(); under2=sum(c[i,j] for i in range(5) for j in range(5) if i-j>=2); over2=sum(c[i,j] for i in range(5) for j in range(5) if j-i>=2)
        out.append([c[2:,2:].sum()/pos, c[:2,:2].sum()/neg, 100*under2/tot, 100*over2/tot, c[2:,:2].sum()])
    return np.array(out)
clin=[]
for run,nm in [("DeiTSmall__opt","DeiT-Small"),("Orig_base__opt","Control"),("Orig_full__opt","KneeOrdinalNet"),("hybrid_mean4","Hybrid")]:
    b=binstats(run,[42,123,2026]); mu=b.mean(0); sd=b.std(0,ddof=1)
    clin.append([nm,"%.3f±%.3f"%(mu[0],sd[0]),"%.3f±%.3f"%(mu[1],sd[1]),"%.0f"%mu[4],"%.1f±%.1f"%(mu[2],sd[2]),"%.1f±%.1f"%(mu[3],sd[3])])
pcs=pd.read_csv(X+"per_class_test.csv"); sensspec=[]
for run,nm in [("DeiTSmall__opt","DeiT-Small"),("Orig_base__opt","Control"),("Orig_full__opt","KneeOrdinalNet"),("hybrid_mean4","Hybrid")]:
    g=pcs[pcs.run==run].groupby("class")[["recall_sensitivity","specificity"]].mean()
    sensspec.append([nm]+["%.2f / %.2f"%(g.loc[k,"recall_sensitivity"],g.loc[k,"specificity"]) for k in ["KL0","KL1","KL2","KL3","KL4"]])
# güven aralıkları (bootstrap, 1000 yeniden örnekleme, yüzdelik; test, seed 42)
bc=pd.read_csv(X+"bootstrap_ci.csv"); ci=[]
for met,nm in [("accuracy","Accuracy"),("balanced_accuracy","Balanced accuracy"),("macro_f1","Macro F1"),("cohen_kappa","Cohen's kappa"),("qwk","QWK"),("mcc","MCC"),("mae_grade","MAE (grades)"),("distant_error_rate","Distant-error rate")]:
    row=[nm]
    for run in ("Orig_full__opt","hybrid_mean4","DeiTSmall__opt"):
        r=bc[(bc.run==run)&(bc.seed==42)&(bc.metric==met)].iloc[0]; row.append("%.3f [%.3f, %.3f]"%(r.value,r.ci_lo,r.ci_hi))
    ci.append(row)
# ek: tüm ağlar için sınıf bazlı metrikler (3 seed ortalaması; tek seedli modellerde seed 42) ve ek metrikler
APX=[("SimpleCNN","SimpleCNN"),("MiniResNet","MiniResNet"),("ResNet50__opt","ResNet50*"),("DenseNet121","DenseNet121"),("EfficientNetB0","EfficientNet-B0"),
     ("MobileNetV3L","MobileNetV3-L"),("Xception","Xception"),("InceptionV3","Inception-v3"),("ConvNeXtTiny__opt","ConvNeXt-Tiny*"),("DeiTSmall__opt","DeiT-Small*"),
     ("Orig_base__opt","Control*"),("Orig_full__opt","KneeOrdinalNet*"),("hybrid_mean4","Hybrid*")]
apx={}
for col,key in [("precision","prec"),("recall_sensitivity","sens"),("specificity","spec"),("f1","f1")]:
    rows=[]
    for run,nm in APX:
        g=pcs[pcs.run==run].groupby("class")[col].mean(); rows.append([nm]+["%.2f"%g.loc[k] for k in ["KL0","KL1","KL2","KL3","KL4"]])
    apx[key]=rows
tr_=pd.read_csv(X+"test_runs_all.csv"); runmap={r:r for r,_ in APX}
extra=[]
for run,nm in APX:
    g=tr_[tr_.run==run]; extra.append([nm,"%.3f"%g.weighted_f1.mean(),"%.3f"%g.qwk.mean(),"%.3f"%g.mae_grade.mean(),"%.3f"%g.adjacent_error_rate.mean(),"%.3f"%g.distant_error_rate.mean(),"%.3f"%g.pr_auc_macro.mean()])
json.dump(dict(abl2=abl2,habl=habl,clin=clin,sensspec=sensspec,ci=ci,apx=apx,extra=extra,main=main,abl=abl,pc=pcrows,hp=hp,tt=tt,cal=cal,pair=pair,total_h=float(res.train_time_s.sum()/3600)),open("data.json","w"),indent=1,ensure_ascii=False)
print(res.train_time_s.sum()/3600)
for r in main: print(r)
for r in abl: print(r)
for r in pcrows: print(r)
for r in hp: print(r)
