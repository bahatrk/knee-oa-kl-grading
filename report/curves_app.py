import pandas as pd, json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R="/home/claude/runs_in/runs_seed_meta/"
M=[("SimpleCNN","SimpleCNN"),("MiniRestNet","MiniResNet"),("ResNet50__opt","ResNet50"),("DenseNet121","DenseNet121"),
("EfficientNetB0","EfficientNet-B0"),("MobileNetV3L","MobileNetV3-L"),("Xception","Xception"),("InceptionV3","Inception-v3"),
("ConvNeXtTiny__opt","ConvNeXt-Tiny"),("DeiTSmall__opt","DeiT-Small"),("Orig_base__opt","Control"),("Orig_full__opt","KneeOrdinalNet")]
plt.rcParams.update({"font.family":"serif","font.size":8})
def draw(kind,fn):
    fig,ax=plt.subplots(4,3,figsize=(7.2,8.4),sharex=False)
    for a,(k,nm) in zip(ax.flat,M):
        h=pd.read_csv(R+k+"/history.csv"); m=json.load(open(R+k+"/meta.json")); b=m["best_epoch"]
        if kind=="loss":
            a.plot(h.epoch,h.train_loss,color="#1f4e79",lw=1.4,label="train"); a.plot(h.epoch,h.val_loss,color="#c55a11",lw=1.4,ls="--",label="validation")
        else:
            a.plot(h.epoch,h.val_macro_f1,color="#1f4e79",lw=1.4)
        a.axvline(b,color="gray",lw=.8,ls=":")
        st=h.stage.values
        for i in range(1,len(st)):
            if st[i]!=st[i-1]: a.axvline(h.epoch.values[i]-.5,color="#999",lw=.6)
        a.set_title(nm,fontsize=8.5); a.grid(alpha=.25)
        for s in("top","right"): a.spines[s].set_visible(False)
    for a in ax[-1]: a.set_xlabel("Epoch")
    for a in ax[:,0]: a.set_ylabel("Loss" if kind=="loss" else "Val macro F1")
    if kind=="loss": ax[0,2].legend(frameon=False,fontsize=7)
    fig.tight_layout(); fig.savefig(fn,dpi=200); plt.close()
draw("loss","figs/f_app_loss.png"); draw("f1","figs/f_app_f1.png")
