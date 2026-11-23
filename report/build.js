const fs = require("fs");
const D = require("docx");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, Footer, AlignmentType, HeadingLevel,
  BorderStyle, WidthType, ShadingType, PageNumber, PageBreak, LevelFormat, PageOrientation, VerticalAlign } = D;
const data = JSON.parse(fs.readFileSync("data.json", "utf8"));
const FONT = "Times New Roman";
const BODY = 22; // 11 pt
const LINE = 276; // 1.15

// ---------- inline markup: **bold**, ~~italic~~ ----------
function runs(text, o = {}) {
  const out = [];
  const parts = text.split(/(\*\*[^*]+\*\*|~~[^~]+~~)/g).filter((s) => s !== "");
  for (const p of parts) {
    if (p.startsWith("**")) out.push(new TextRun({ text: p.slice(2, -2), bold: true, font: FONT, size: o.size || BODY, italics: o.italics }));
    else if (p.startsWith("~~")) out.push(new TextRun({ text: p.slice(2, -2), italics: true, font: FONT, size: o.size || BODY, bold: o.bold }));
    else out.push(new TextRun({ text: p, font: FONT, size: o.size || BODY, bold: o.bold, italics: o.italics }));
  }
  return out;
}
const P = (t, o = {}) => new Paragraph({ children: runs(t, o), alignment: o.align || AlignmentType.JUSTIFIED, spacing: { line: LINE, after: o.after ?? 100, before: o.before ?? 0 }, keepNext: o.keepNext, indent: o.indent });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })], keepNext: true });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })], keepNext: true });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: t, font: FONT })], keepNext: true });
const BUL = (t) => new Paragraph({ numbering: { reference: "bul", level: 0 }, children: runs(t), alignment: AlignmentType.JUSTIFIED, spacing: { line: LINE, after: 60 } });
const NUM = (t) => new Paragraph({ numbering: { reference: "num", level: 0 }, children: runs(t), alignment: AlignmentType.JUSTIFIED, spacing: { line: LINE, after: 60 } });

// ---------- figures ----------
function pngSize(f) { const b = fs.readFileSync(f); return [b.readUInt32BE(16), b.readUInt32BE(20)]; }
function FIG(file, widthIn, label, caption) {
  const [w, h] = pngSize("figs/" + file);
  const W = Math.round(widthIn * 96), Hh = Math.round((W * h) / w);
  return [
    new Paragraph({ children: [new ImageRun({ type: "png", data: fs.readFileSync("figs/" + file), transformation: { width: W, height: Hh }, altText: { title: label, description: caption, name: label } })],
      alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true }),
    new Paragraph({ children: [new TextRun({ text: label + ". ", bold: true, font: FONT, size: 19 }), ...runs(caption, { size: 19 })], alignment: AlignmentType.CENTER, spacing: { line: 240, after: 160 } }),
  ];
}
// ---------- tables ----------
const border = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };
function TBL(widths, rows, o = {}) {
  const fs_ = o.size || 18;
  const total = widths.reduce((a, b) => a + b, 0);
  const mk = (r, i) => new TableRow({
    tableHeader: i === 0, cantSplit: true,
    children: r.map((c, j) => new TableCell({
      borders, width: { size: widths[j], type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
      shading: i === 0 ? { fill: "DCE6F2", type: ShadingType.CLEAR, color: "auto" } : (o.shade && o.shade(r, i) ? { fill: "F3F3F3", type: ShadingType.CLEAR, color: "auto" } : undefined),
      margins: { top: 40, bottom: 40, left: 70, right: 70 },
      children: [new Paragraph({ children: runs(String(c), { size: fs_, bold: i === 0 }), alignment: j === 0 || o.left ? AlignmentType.LEFT : AlignmentType.CENTER, spacing: { line: 240, after: 0 } })],
    })),
  });
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: rows.map(mk) });
}
const TCAP = (label, caption) => new Paragraph({ children: [new TextRun({ text: label + ". ", bold: true, font: FONT, size: 19 }), ...runs(caption, { size: 19 })], alignment: AlignmentType.LEFT, spacing: { line: 240, before: 140, after: 80 }, keepNext: true });
const GAP = () => new Paragraph({ children: [], spacing: { after: 120 } });

// ---------- numbering constants ----------
const T = { lit: 1, audit: 2, rat: 3, space: 4, log: 5, main: 6, pc: 7, clin: 8, ci: 9, cal: 10, abl: 11, pair: 12, habl: 13, corr: 14, sel: 15, hp: 16, tt: 17, ablx: 18, apxp: 19, apxs: 20, apxk: 21, apxf: 22, apxe: 23 };

// ================= CONTENT =================
const cover = [
  new Paragraph({ children: [], spacing: { before: 1800 } }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [new TextRun({ text: "Grading Knee Osteoarthritis Severity on Radiographs with a Dataset-Specific Ordinal Network", bold: true, font: FONT, size: 36 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 }, children: [new TextRun({ text: "Comparison with transfer-learning models and a hybrid ensemble, explainability, calibration and robustness analysis on the kneeKL224 dataset", italics: true, font: FONT, size: 26 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "Baha Türk", bold: true, font: FONT, size: 28 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 }, children: [new TextRun({ text: "Student number: 220709071", font: FONT, size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "MBM6 453 Introduction to Artificial Intelligence – Course Project 2026–2027", font: FONT, size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "Assigned dataset: VS13 – Knee Osteoarthritis Severity Grading Dataset (Kellgren–Lawrence, 5 classes)", font: FONT, size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "November 2026", font: FONT, size: 24 })] }),
];

const abstract = [
  H1("Abstract"),
  P(`Knee osteoarthritis (OA) severity is graded on radiographs with the five-level Kellgren–Lawrence (KL) scale, a task limited by subjective reading and by the subtle boundary between KL1 and KL2. Using the kneeKL224 dataset (8,260 radiographs; official patient-disjoint split of 5,778/826/1,656 images), we compare a HOG+SVM baseline with twelve neural networks and propose KneeOrdinalNet, a ConvNeXt-Tiny network with four data-driven components: per-image intensity standardization, multi-scale average-and-max pooling, a joint-line row-attention module and a Gaussian-soft-label ordinal loss. All selections used validation macro F1 only, and the test set was evaluated once. Averaged over three seeds, KneeOrdinalNet reached a macro F1 of 0.713 ± 0.008, a quadratic weighted kappa (QWK) of 0.862 and a mean absolute grade error of 0.332, against 0.698, 0.840 and 0.363 for the best existing model (DeiT-Small). The macro-F1 difference was significant in only one of three seeds, and a control with the same hyperparameters but no custom components scored 0.707, so most of the gain comes from tuning, while the gain in ordinal agreement is more consistent. Calibration was worse, KL1 recall was lower, and a four-model ensemble (0.717) added little at 3.6 times the parameters. Grad-CAM, Integrated Gradients, corruption-robustness and selective-prediction analyses are reported. The model is not suitable for clinical use.`),
  P(`**Keywords:** knee osteoarthritis; Kellgren–Lawrence grading; ordinal classification; deep learning; explainability; calibration`, { align: AlignmentType.LEFT }),
];

const intro = [
  H1("1. Introduction"),
  P(`Knee osteoarthritis is a common degenerative joint disease and a major cause of pain and reduced mobility in older adults. Plain radiography remains the first-line imaging method, and disease severity is most often summarized with the Kellgren–Lawrence (KL) scale [1], which assigns grades from KL0 (no radiographic evidence) to KL4 (severe) according to osteophytes, joint-space narrowing, sclerosis and bone deformity. The grade influences follow-up and referral decisions, so a consistent reading is valuable. In practice it is not: agreement between readers is only moderate, with a kappa of about 0.6 reported between radiologists and surgeons [19], and reliability improves when the five grades are collapsed into a coarse "KL ≤ 2 versus KL ≥ 3" decision [20]. The hardest distinction is at the low end, where doubtful (KL1) and minimal (KL2) changes overlap.`),
  P(`Deep learning has been applied to automatic KL grading since 2016 [2–5], and recent work reports ever higher accuracies [7–17]. Three shortcomings recur. First, many studies optimize and report accuracy alone, although the KL grades are ordered and an error of three grades is clinically very different from an error of one grade. Second, data handling is often under-specified: duplicate images, patient overlap between splits and the role of the test set in model selection are rarely described, which makes numbers hard to compare. Third, calibration and the reliability of saliency maps are seldom examined, although both determine whether a model's output can be trusted by a reader.`),
  P(`This study uses the public kneeKL224 collection (8,260 images, 224 × 224 pixels, official train/validation/test split) and asks: **can a small set of architectural components motivated by concrete properties of this dataset improve KL grading over the strongest existing transfer-learning model, when both are tuned and evaluated under the same leakage-free protocol, and what do these components cost in calibration, robustness and efficiency?** Besides the main comparison, we ask whether the answer survives a matched control and multiple seeds, and whether explanations of the model can be trusted.`),
  P(`The main contributions are:`, { after: 40, keepNext: true }),
  BUL(`**An original model, KneeOrdinalNet** (Section 3.7), a ConvNeXt-Tiny network with four components tied to dataset properties (exposure variation, multi-scale evidence, horizontal joint line, ordered labels). The design is justified by a rationale table, a four-version design log, four ablations repeated with three seeds, an exact McNemar test against the component that contributes most, and a matched control with identical hyperparameters.`),
  BUL(`**A leakage-aware, validation-only protocol**: file, duplicate and patient-level audit (perceptual hashing with pixel-level verification); Optuna search for four models; three seeds; bootstrap confidence intervals and paired tests; and a single evaluation of the test set after all choices were frozen.`),
  BUL(`**An evaluation that goes beyond accuracy**: ordinal metrics (QWK, mean absolute grade error, adjacent and distant errors), calibration with temperature scaling, a hybrid ensemble with four fusion methods, saliency analysis with two attribution methods plus randomization and deletion tests, and two challenge tasks (corruption robustness and selective prediction).`),
  BUL(`**Honest reporting of negative findings**: most of the macro-F1 gain over a default pipeline is explained by hyperparameter tuning; the ordinal loss harms calibration; recall for KL1 is lower than for the best existing model; and the ensemble is not worth its cost.`),
];

const litRows = [
  ["Study", "Data", "Method", "Reported result", "Strength", "Limitation noted"],
  ["Antony et al. 2016 [2]", "OAI knees", "CNN, regression loss", "Accuracy 59.6%", "Showed that a regression loss helps; argued for distance-based metrics", "Early baseline; nominal metrics only"],
  ["Tiulpin et al. 2018 [3]", "OAI, held-out test", "Siamese CNN", "QWK 0.83; accuracy 66.7%", "Trained on MOST, tested on OAI (independent cohort); attention maps; code released", "Needs symmetric knee pairs; reader agreement is similar"],
  ["Chen et al. 2019 [4]", "OAI; source of kneeKL224", "YOLOv2 detection + VGG-19, ordinal loss", "Accuracy 69.7%; MAE 0.344", "Ordinal loss; released the cropped data (kneeKL224)", "Two-stage pipeline; no calibration"],
  ["Pi et al. 2023 [7]", "kneeKL224", "Six-model ensemble", "Accuracy 76.9%; F1 0.767", "Same data as ours; input size chosen per member", "Protocol unclear (pooled train/val CV); KL1 F1 only 0.45"],
  ["Tariq et al. 2023 [8]", "Own 7:2:1 split", "Ensemble of four CNNs", "Accuracy 98%; QWK 0.99", "Several CNNs compared before ensembling", "Very high, different split; leakage cannot be excluded"],
  ["Yoon et al. 2023 [9]", "OAI train, 400 test knees", "Feature extraction + KL grading", "Accuracy 0.83; kappa 0.768", "Grades individual features (joint-space narrowing, osteophytes), not only KL", "Different data and test size; not directly comparable"],
  ["Jahan et al. 2024 [10]", "Four merged datasets, 110,232 images (DCGAN-augmented)", "Compact convolutional transformer", "Accuracy 94.6%", "Very large training set; compact transformer", "Synthetic images and merged sources; split effect unclear"],
  ["Kılıç 2025 [12]", "13,254 images", "DenseNet201 + multi-scale attention", "Accuracy 94.7%", "Multi-scale attention, related to our component B", "Data and split description unclear"],
  ["Qiu et al. 2025 [13]", "Not named on the article page", "Dual vision transformer", "Accuracy 78.4%; MAE 0.471", "Reports MAE, an ordinal error measure", "Dataset and split not stated"],
  ["Ullah and Kim 2026 [14]", "kneeKL224 test set (1,656)", "ViT-L/16, CORAL, CLIP distillation, 5-fold ensemble, TTA", "Accuracy 70.3%; macro F1 70.2%", "Official test split; gains on KL1/KL2; code released", "Very heavy; no QWK or MAE"],
  ["Kizilkaya et al. 2026 [15]", "8,260 images", "VGG-16 + CORAL", "QWK 0.830; macro F1 0.676", "Rank-consistent ordinal head; QWK and per-class AP reported", "KL1 weakest (AP 0.337)"],
  ["Khatami et al. 2026 [16]", "8,260 images", "ResNet-18 ordinal regression", "Accuracy 67.8%; QWK 0.839; F1 0.675", "Compares four ordinal strategies on the predefined split", "Single model, single run"],
];
const lit = [
  H1("2. Literature Review"),
  P(`**Why the task is hard.** The KL scale was proposed in 1957 [1] and relies on verbal descriptions ("doubtful" narrowing, "possible" osteophytes), which explains the modest inter-reader agreement [19]. A recent dataset-construction study found that reliability on five grades was poor but improved when KL ≤ 2 and KL ≥ 3 were separated [20]. Any model trained on such labels inherits their noise, most visibly around KL1. This motivates (i) ordinal losses that tolerate neighbouring-grade errors and (ii) metrics that distinguish adjacent from distant errors.`),
  P(`**Early deep-learning work.** Antony et al. fine-tuned CNNs on knee radiographs and showed that a regression-style loss beats plain classification [2]. Tiulpin et al. used a Siamese network exploiting knee symmetry and reported a QWK of 0.83 [3]. Chen et al. localized the joint with YOLOv2 and graded it with VGG-19 trained with an ordinal loss; the cropped data they released is the kneeKL224 collection used here [4]. Thomas et al. then trained an end-to-end model on full radiographs that reached performance comparable to individual musculoskeletal radiologists and showed saliency maps [5], and detection-plus-transformer pipelines on Osteoarthritis Initiative data appeared later [18]. These studies established the ordinal view of the problem but reported few seeds, little uncertainty, and no calibration.`),
  P(`**Recent studies (2022–2026).** Table ${T.lit} compares twelve studies by data, method, result, strength and limitation. Four patterns emerge. (1) ~~Ordinal modelling~~ is now standard: an ordinal regression module [6], CORAL layers [14,15,28] and ordinal regression with a small ResNet [16] all give a QWK around 0.83–0.84 on the full-resolution-free kneeKL224 images. (2) ~~Ensembling and scale~~ buy accuracy: a six-model ensemble [7] and a ViT-L/16 five-fold ensemble with test-time augmentation [14] reach 70–77% accuracy, at large computational cost. (3) ~~Very high accuracies~~ (94–98%) [8,10,12] come from merged datasets, synthetic augmentation or splits that are not described well enough to exclude leakage, and cannot be compared with studies that use the official split. (4) ~~Clinical-style systems~~ [9] report agreement with specialists (kappa 0.768) on much larger OAI-based training sets, a different problem setting. Plug-in modules for fine-grained discrimination were trained on Osteoarthritis Initiative data and tested on an independent cohort; the authors again report KL1 as the least accurate grade (43%) [11]. Newer work also studies uncertainty, for example evidential ordinal regression in a preprint that has not been peer reviewed [17], and the reliability of KL labels [19,20]. Reports on kneeKL224 consistently find KL1 the weakest class [7,15].`),
  P(`**Gap addressed here.** Few studies combine, on one public split, (a) an architecture designed from dataset properties and tested with ablations and a matched control, (b) multiple seeds and paired significance tests, (c) calibration and selective prediction, and (d) validity checks of the saliency maps (randomization and deletion tests [24–26]). This article does so, and reports where the proposed design fails to help.`),
  P(`**Figure conventions in similar articles.** Articles in this area typically present a pipeline flowchart and an architecture diagram with tensor sizes, training and validation curves with the stopping epoch, row-normalized confusion matrices (rows are true grades), ROC and precision–recall curves, and Grad-CAM overlays for correct and incorrect cases. We follow these conventions and add reliability diagrams, deletion curves and risk–coverage curves, so that the figures support the claims made about reliability, not only accuracy.`),
  TCAP(`Table ${T.lit}`, "Selected knee-OA grading studies. Results are as reported by the authors and are not directly comparable."),
  TBL([1250, 1250, 1550, 1400, 1800, 1822], litRows, { size: 16, left: true }),
  GAP(),
];

const auditRows = [
  ["Item", "Finding"],
  ["Files", "8,260 listed, 8,260 readable, 0 corrupted; split sizes 5,778 / 826 / 1,656 match the project brief"],
  ["Images", "All 224 × 224, 8-bit grayscale; mean pixel intensity 154.9 (mean within-image SD 38.5); 136 intensity outliers (robust z > 3.5), kept"],
  ["Class balance", "Train KL0–KL4: 2,286 / 1,046 / 1,516 / 757 / 173; largest-to-smallest ratio 13.2 (validation 12.2, test 12.5)"],
  ["Exact duplicates", "None (pixel MD5)"],
  ["Near duplicates", "pHash (64 bit), Hamming ≤ 4: 498 candidate pairs (711 images), 222 across splits, 125 train–test; 321 pairs carry different labels, i.e. visually similar but different knees. Pixel-level verification (SSIM ≥ 0.90 and NCC ≥ 0.95) confirmed only 4 duplicate pairs, 2 across splits"],
  ["Patients", "Patient ID parsed from 100% of file names (7-digit ID + L/R): 4,130 patients, 2 knees each; 2,889 / 413 / 828 patients per split; **zero patients in more than one split**"],
];
const spaceRows = [
  ["Hyperparameter", "Search range"],
  ["Head learning rate / fine-tuning learning rate", "log-uniform [3e-4, 3e-3] / log-uniform [2e-5, 5e-4]"],
  ["Weight decay", "log-uniform [1e-4, 1e-1]"],
  ["Label smoothing / dropout", "uniform [0, 0.15] / uniform [0.1, 0.5]"],
  ["Fraction of backbone unfrozen / batch size", "{0.25, 0.5, 0.75, 1.0} / {32, 64}"],
  ["Ordinal loss σ and λ (KneeOrdinalNet only)", "added to the space above"],
];
const ratRows = [
  ["Component", "Dataset property targeted", "Ablation that tests it"],
  ["A. Per-image standardization", "Exposure differences: wide brightness distribution and 136 dark, bright or low-contrast outliers", "Remove A (version v3); contrast-loss corruption test"],
  ["B. Multi-scale average + max pooling over stages 2–4", "Evidence at two scales: small osteophytes (fine detail, shallow stages, max pooling) and joint-space narrowing (broad structure, deep stage)", "Replace B by last-stage average pooling (noMS)"],
  ["C. Joint-line row attention", "Key evidence lies on the horizontal femur–tibia line whose vertical position varies with positioning; row gating is independent of lateral position", "Remove C (noRow); exact McNemar test against the full model"],
  ["D. Ordinal loss (Gaussian soft labels + expected-grade smooth-L1)", "Ordered labels, mostly adjacent errors, costly distant errors, noisy KL1 boundary, 13:1 imbalance", "Replace D by cross-entropy (noOrd)"],
  ["Control", "Does any gain come from design or from tuning?", "Plain ConvNeXt-Tiny, none of A–D, identical hyperparameters"],
];
const logRows = [
  ["Version", "Components", "Val macro F1", "Val distant errors", "Reason for the next step"],
  ["v1", "B", "0.698", "5.4%", "Baseline of the custom family; add attention on the joint line (C)"],
  ["v2", "B + C", "0.705", "4.7%", "F1 rose, but distant errors remain; add the ordinal loss (D)"],
  ["v3", "B + C + D", "0.694", "3.8%", "Distant errors fell; add exposure invariance (A)"],
  ["Full", "A + B + C + D", "0.689", "3.4%", "Lowest distant errors and highest QWK (0.831); kept, then tuned with Optuna"],
];

const methods = [
  H1("3. Materials and Methods"),
  H2("3.1 Dataset, license and ethics"),
  P(`The data are the Knee Osteoarthritis Severity Grading Dataset (folder kneeKL224) from Mendeley Data [29] (https://data.mendeley.com/datasets/56rmx5bjcr/1, DOI 10.17632/56rmx5bjcr.1), released under CC BY 4.0 and derived from the Osteoarthritis Initiative radiographs processed as in [4]. It contains 8,260 knee images with KL grades 0–4. The archive also holds a 299 × 299 version of the images (kneeKL299), an additional test folder and pretrained model files; none of these were used, only the train, val and test folders of kneeKL224. The official split (5,778 training, 826 validation, 1,656 test images) was used unchanged. The data were used only for education and research; no attempt was made to identify patients, the raw images are not redistributed, and the developed model is **not** a clinical decision tool. The dataset page and publication are cited as required by the license.`),
  H2("3.2 Data audit and splitting"),
  P(`The audit (Table ${T.audit}) checked file integrity, sizes, intensity statistics, class balance, duplicates and patient identifiers. Near duplicates were screened with a 64-bit perceptual hash; the Hamming-distance threshold of 4 flagged 498 pairs, but 321 of them had different labels and most were different knees that merely look alike. A second, pixel-level check (SSIM and normalized cross-correlation) confirmed only four true duplicate pairs, two of them across splits. We kept the official split and ran a sensitivity analysis that excludes the confirmed duplicates; test macro F1 was identical to three decimals. File names encode a seven-digit patient number and the side, so patient-level splitting could be verified: the official split is patient-disjoint (2,889, 413 and 828 patients; no overlap), and each patient contributes one image per knee. The class ratio of 13.2 exceeds the threshold of 3, so a balancing method was mandatory (Section 3.5).`),
  TCAP(`Table ${T.audit}`, "Data audit summary."),
  TBL([1700, 7372], auditRows, { size: 17, left: true }),
  GAP(),
  ...FIG("f_flow.png", 6.3, "Figure 1", "Project flowchart. Blue steps used validation data; orange steps were run on the test set after all choices were frozen."),
  ...FIG("f_classdist.png", 6.3, "Figure 2", "Images per class and class share in each split (hatches distinguish splits)."),
  ...FIG("f_samples.png", 5.4, "Figure 3", "Random training images for each KL grade (seed 42)."),
  P(`Figures 2 and 3 show the imbalance (KL4 is rare) and how little the grades differ visually at the low end. Figure 4 shows the exposure variation that motivated component A.`),
  ...FIG("f_intensity.png", 5.4, "Figure 4", "Distribution of mean brightness and contrast per class."),
  H2("3.3 Preprocessing and augmentation"),
  P(`The radiographs are grayscale; they were kept at their native 224 × 224 resolution and replicated to three channels. Networks with ImageNet weights used the normalization of their own pretrained configuration (read from timm); the two networks trained from scratch used the mean and standard deviation of a random sample of 1,500 training images, so no statistic was computed from validation or test data; KneeOrdinalNet additionally standardizes every image by its own statistics (component A). The same 224 × 224 input was used for all models, including Xception and Inception-v3 whose native size is 299 × 299, so that the computing settings are identical across models; this may disadvantage these two networks slightly. Training augmentation consisted of horizontal flips (p = 0.5), small affine transforms (rotation up to 7°, translation up to 5%, scale 0.93–1.07) and brightness and contrast jitter of ±20%; these keep the joint anatomy realistic. Validation and test images were not augmented. Figure 5 shows originals and augmented versions.`),
  ...FIG("f_aug.png", 5.6, "Figure 5", "Original training image (left) and four random augmentations for KL0, KL2 and KL4."),
  H2("3.4 Classical baseline and standard networks"),
  P(`The classical baseline is a histogram-of-oriented-gradients (HOG) descriptor (9 orientations, 32-pixel cells, 2 × 2 blocks) with a class-balanced linear SVM (C = 0.01, chosen on validation). The twelve neural networks follow the required distribution: two CNNs trained from scratch (SimpleCNN, MiniResNet); seven CNNs with ImageNet weights (ResNet50, DenseNet121, EfficientNet-B0, MobileNetV3-Large, Xception, Inception-v3 and ConvNeXt-Tiny [21]); one attention-based model (DeiT-Small [22]); the original model KneeOrdinalNet; and the hybrid ensemble (Section 3.8). A matched control (plain ConvNeXt-Tiny with the hyperparameters of KneeOrdinalNet) was trained in addition and is not counted among the twelve. ConvNeXt-Tiny, DeiT-Small and ResNet50 were the three best candidates after a screening round on validation macro F1 and received the full hyperparameter search.`),
  H2("3.5 Training protocol and class imbalance"),
  P(`All networks were trained with AdamW, cosine learning-rate decay, mixed precision, label smoothing and dropout. Pretrained networks first trained the new head for four epochs and then fine-tuned a fraction of the backbone for up to 20 epochs; scratch networks trained for up to 40 epochs. Early stopping used validation macro F1 with a patience of six epochs, and the best-epoch weights were restored. To handle imbalance, four options were compared on validation with EfficientNet-B0 (seed 42): plain cross-entropy (macro F1 0.612), class-weighted cross-entropy (0.599), focal loss (0.609) and cross-entropy with a weighted random sampler (0.622). The sampler was best and used for all models; the unbalanced version was not better.`),
  H2("3.6 Hyperparameter search"),
  P(`For ConvNeXt-Tiny, DeiT-Small, ResNet50 and KneeOrdinalNet we ran 25 Optuna [27] trials each with the TPE sampler (seed 42) and a median pruner, using only the training and validation data. To limit cost, each trial used a stratified 40% subset of the training set and a reduced budget (3 head epochs, 10 fine-tuning epochs, patience 4). The objective was validation macro F1 at the best epoch. Pruning stopped 19 trials for ResNet50, 15 for ConvNeXt-Tiny, 18 for DeiT-Small and 17 for KneeOrdinalNet. Best values were then retrained on the full training set with the full budget (Table ${T.space}; best values in Appendix A). The control uses the hyperparameters found for KneeOrdinalNet, which, if anything, favours the control relative to its own search.`),
  TCAP(`Table ${T.space}`, "Optuna search space."),
  TBL([4200, 4872], spaceRows, { size: 17, left: true }),
  GAP(),
  H2("3.7 Original model: KneeOrdinalNet"),
  P(`**Problem analysis.** Three properties of kneeKL224 shaped the design: (i) exposure varies strongly between images; (ii) the decisive evidence is a thin horizontal structure, the femorotibial joint line, plus small osteophytes at the margins, so both fine detail and broad structure matter; (iii) the labels are ordered, mostly confused with a neighbour, and the KL1 boundary is noisy. KneeOrdinalNet therefore keeps the pretrained ConvNeXt-Tiny backbone (28.4 M parameters in total) and adds four components (Table ${T.rat}, Figure 6).`),
  TCAP(`Table ${T.rat}`, "Design rationale: component, targeted data property and the ablation that tests it."),
  TBL([2300, 4200, 2572], ratRows, { size: 16, left: true }),
  GAP(),
  ...FIG("f_arch.png", 6.3, "Figure 6", "KneeOrdinalNet: backbone stem and four stages with output sizes (channels × height × width) and the three branches that end in the 2688-dimensional vector. Orange: new components A–C; D is a training loss. Parameters per block."),
  P(`**Components.** (A) The input of each image is standardized by its own mean and standard deviation, which makes the network invariant to brightness and contrast changes (the dataset-level normalization is affine per channel, so this equals standardizing the raw image). (B) Feature maps of stages 2, 3 and 4 (192 × 28 × 28, 384 × 14 × 14 and 768 × 7 × 7) are pooled by both average and max, normalized with LayerNorm and concatenated into a 2,688-dimensional vector (multi-scale fusion in the spirit of feature pyramids [33], here applied to pooled features) (13,445 parameters in the linear head). (C) Before pooling, each stage passes through a row attention module: it summarizes each row of the feature map by its width-wise mean and max, applies two one-dimensional convolutions along the vertical axis (a gate in the style of squeeze-and-excitation [30] and CBAM [31] that acts on rows instead of channels or pixels; the row design is our own adaptation) to include neighbouring rows, and produces a gate 2·sigmoid(z) per row. The last convolution is zero-initialized so that the gate starts as the identity and does not disturb pretrained features; the modules add 27,745, 110,785 and 442,753 parameters. Because the summary runs along the width, the gate does not depend on where the knee lies laterally. (D) The loss replaces the one-hot target by a Gaussian soft label with standard deviation σ over grades (soft ordinal labels as in [32]), plus λ times a smooth-L1 term between the expected grade and the true grade, so that distant errors are penalized more than adjacent ones; σ = 0.617 and λ = 0.094 after tuning. In total 590,723 parameters are added to the backbone (2.1%).`),
  P(`**Design log.** Four versions were built in sequence (Table ${T.log}); all used the same untuned hyperparameters and seed 42. The validation macro F1 did not increase monotonically: the largest value (0.705) belongs to v2, and the final version had the lowest macro F1 (0.689) but the lowest distant-error rate and the highest QWK. We kept the final design because its aim is ordinal consistency, which we state openly as a judgement call: it was not the macro-F1 maximum. After tuning, the full model reached a validation macro F1 of 0.703 versus 0.699 for the control (Section 4.5).`),
  TCAP(`Table ${T.log}`, "Design log, validation set, seed 42, untuned hyperparameters."),
  TBL([800, 1400, 1100, 1300, 4472], logRows, { size: 17, left: true }),
  GAP(),
  H2("3.8 Hybrid model"),
  P(`The hybrid combines four trained, tuned members from different families, KneeOrdinalNet (convolutional with custom components), DeiT-Small (transformer), ConvNeXt-Tiny and ResNet50, at the probability level, using seed-matched members so that results are again mean ± SD over three seeds. Four fusion methods were compared on validation: equal-weight mean of probabilities; weighted mean with weights chosen on a 0.1-step simplex to maximize macro F1; stacking with multinomial logistic regression on log-probabilities; and the mean followed by an ordinal decision rule that picks the grade minimizing expected absolute grade error. Weights and meta-learner were learned on validation data only, and their validation scores were computed out-of-fold (five folds) to avoid optimism. In total 44 member-set and method combinations were compared; the choice was the equal-weight mean of all four members. Test-time use costs the sum of the members' cost.`),
  H2("3.9 Evaluation protocol and statistics"),
  P(`Models, epochs, thresholds and ensemble weights were selected only by validation macro F1. The test set was predicted once per trained model, the probabilities were cached, and a manifest with the hash of each model file was stored; all test tables derive from those cached probabilities. Reported metrics are accuracy, balanced accuracy, macro precision, macro recall, macro F1, Cohen's kappa, Matthews correlation coefficient, macro one-vs-rest ROC AUC, quadratic weighted kappa (QWK), mean absolute grade error (MAE), and adjacent (|error| = 1) and distant (|error| ≥ 2) error rates. Seeds 42, 123 and 2026 changed initialization, data order and augmentation; the split stayed fixed. We report mean ± SD over seeds, 95% percentile bootstrap intervals (1,000 resamples), paired bootstrap tests for differences between models on the same test images (1,000 resamples), and an exact McNemar test for the ablation that mattered most on validation. Calibration was measured by the expected calibration error (ECE, 15 equal-width bins) and the Brier score; temperature scaling [23] was learned on validation only. CPU latency is the median time for a single image (batch size 1) on a CPU; model size is the weight file size.`),
  H2("3.10 Explainability and challenge protocols"),
  P(`For KneeOrdinalNet and DeiT-Small we computed Grad-CAM [24] (last convolutional stage; last transformer block reshaped to the patch grid; pytorch-grad-cam [37]) and Integrated Gradients [25] (Captum [36]). Because KneeOrdinalNet standardizes each image, zero or scaled baselines are meaningless for it; a blurred copy of the image was used as baseline. For DeiT-Small, attention rollout was added. We explained 20 correctly classified test images (four per grade) and 12 misclassified ones (distant errors first). Reliability of the maps was tested with the model-parameter randomization test [26] (layer groups re-initialized cumulatively from output to input; Spearman similarity to the original Grad-CAM) and with deletion curves (most important pixels replaced by a blurred value; 60 class-balanced images; maps are computed for the true grade and the probability of the true grade is tracked). We also measured the share of each map's total attribution that falls into the outer 12-pixel band of the input (20.3% of its area), where frames, text and device traces would appear. The two challenge tasks were (1) robustness to Gaussian noise, Gaussian blur, JPEG compression and contrast loss at five severities, and (2) selective prediction: images with the lowest confidence are rejected, with confidence thresholds set on validation to reach target coverages, and the risk–coverage curve is reported.`),
  H2("3.11 Hardware and reproducibility"),
  P(`All training ran on Google Colab with a single NVIDIA T4 GPU (16 GB). The summed logged training time of the final runs was ${data.total_h.toFixed(1)} hours (Appendix C), excluding hyperparameter-search trials and evaluations. The operating system was Linux (kernel 6.6.159+, glibc 2.39) with Python 3.13.18. Training used PyTorch 2.11.0 [34] (CUDA 13.0 build) and torchvision 0.26.0, timm 1.0.30 [35], NumPy 2.4.6, pandas 2.2.3, scikit-learn 1.6.1 [38], SciPy 1.16.3, scikit-image 0.25.2 and Pillow 11.3.0; the search and explainability libraries were pinned to Optuna 5.0.0, imagehash 4.3.2, Grad-CAM 1.5.7 and Captum 0.9.0. The data audit ran on a CPU runtime (PyTorch 2.11.0+cpu), and the additional latency measurements of Table ${T.abl} ran on a CPU runtime (Intel Xeon at 2.20 GHz, two virtual CPUs) with PyTorch 2.14.1+cpu and one thread; the library versions of the session that produced the latencies of Table ${T.main} were not recorded separately. All version listings are stored with the code. Random seeds were set for Python, NumPy and PyTorch; GPU nondeterminism can still cause small differences. The code repository contains the commands to reproduce every table and figure, the dataset link and archive hash, a single-image CPU prediction script and the saved weights.`),
];

const resultsMain = [
  H1("4. Results"),
  H2("4.1 Main comparison"),
  P(`Table ${T.main} (next page) lists all test results. Among single models, KneeOrdinalNet has the best values: macro F1 0.713 ± 0.008, accuracy 0.694, kappa 0.581, MCC 0.582 and macro AUC 0.911. The best existing model, selected on validation, is DeiT-Small (0.698 ± 0.007). ConvNeXt-Tiny reaches 0.686, ResNet50 0.660, and the scratch networks 0.607 and 0.631. The classical HOG+SVM baseline reaches only 0.376 (95% CI 0.345–0.405) and a QWK of 0.581, confirming that deep features matter. Hyperparameter search helped every pretrained model: macro F1 rose from 0.674 to 0.686 for ConvNeXt-Tiny, from 0.660 to 0.698 for DeiT-Small and from 0.637 to 0.660 for ResNet50. The bootstrap intervals of single seeds are about ±0.023 wide (for example, 0.681–0.726 for KneeOrdinalNet and 0.675–0.726 for DeiT-Small, both seed 42), so the 0.015 difference between them is within sampling noise for a single test set.`),
];
const mainTable = [
  TCAP(`Table ${T.main}`, `Test-set results. Mean ± SD of three seeds for models marked *, single seed (42) otherwise. Bold: best value in the column (efficiency columns: best neural network). Latencies are from the first timing session; the control and ablation variants were timed in a second session (Table ${T.abl}, single CPU thread, higher absolute values), so compare latencies only within a table.`),
  TBL([1750, 1350, 650, 650, 950, 950, 900, 900, 950, 950, 950, 950, 600, 1150],
    [["Model", "Family", "Input", "Params (M)", "Acc.", "Bal. acc.", "Macro prec.", "Macro rec.", "Macro F1", "Kappa", "MCC", "Macro AUC", "Size (MB)", "CPU lat. med. / p95 (ms)"], ...data.main],
    { size: 14, shade: (r) => r[0].startsWith("KneeOrdinalNet") }),
];
const resultsRest = [
  H2("4.2 Training curves and confusion matrices"),
  P(`Figure 7 shows training and validation curves for the best existing model and KneeOrdinalNet (seed 42). DeiT-Small reached its best epoch at 22 of 24 and was stopped by the epoch budget, not by early stopping, so it might gain slightly from longer training; KneeOrdinalNet reached its best epoch at 13 and was stopped at epoch 19 by the patience rule, so its curve is shorter. Both models fit the training set much better (accuracy about 0.81–0.84) than the validation set (0.65–0.67); this gap of about 15–18 points shows overfitting that label smoothing, dropout and augmentation reduce but do not remove. The bump at epoch 5 marks the moment the backbone is unfrozen. Validation curves are noisy, which is expected with 826 images and a rare KL4 class (27 images).`),
  ...FIG("f_curves.png", 5.8, "Figure 7", "Loss and accuracy of DeiT-Small and KneeOrdinalNet, training and validation sets, seed 42, same epoch range. Black dotted line: best epoch; red: end of training. Both models fit training data far better than validation data."),
  P(`Figure 8 shows row-normalized confusion matrices on the test set (rows: true grade; columns: predicted grade; values are fractions of each true class; seed 42). Errors are overwhelmingly to neighbouring grades. KneeOrdinalNet recovers KL0 (0.79), KL3 (0.84) and KL4 (0.88) better than DeiT-Small (0.76, 0.82, 0.78), and KL2 (0.64 versus 0.59), but its KL1 recall is lower (0.39 versus 0.53): 35% of KL1 knees are called KL0 and 25% KL2. The hybrid is close to DeiT-Small on KL1 (0.53). Three-seed means show the same pattern (Table ${T.pc}).`),
  ...FIG("f_conf.png", 6.3, "Figure 8", "Row-normalized test confusion matrices, seed 42 (true grade in rows)."),
  TCAP(`Table ${T.pc}`, "Per-class sensitivity / specificity on the test set (one-versus-rest), mean of three seeds."),
  TBL([1500, 1514, 1514, 1514, 1515, 1515], [["Model", "KL0", "KL1", "KL2", "KL3", "KL4"], ...data.sensspec], { size: 17 }),
  GAP(),
  P(`**Clinical perspective.** For a screening or triage aid the costlier error is missing definite radiographic osteoarthritis, that is, calling a knee KL0 or KL1 when its true grade is KL2 or higher (a false negative at the usual KL ≥ 2 threshold). It can delay referral, advice on load and weight, and follow-up, whereas the opposite error mainly causes an additional review. Within the grading task, under-grading by two or more grades is the worst case. Table ${T.clin} derives sensitivity and specificity for the KL ≥ 2 decision from the five-class predictions, without any threshold tuned on the test set. KneeOrdinalNet misses on average 141 of the 721 test knees with KL ≥ 2 (sensitivity 0.805 ± 0.016), against 158 for DeiT-Small (0.780 ± 0.027) and 167 for the control (0.768 ± 0.034), at a comparable specificity (0.918 versus 0.919 and 0.934). These differences are of the order of one seed SD and therefore indicative only. For the individual grades (Table ${T.pc}), the sensitivity for KL1, the doubtful grade, is low for all models (0.43–0.53) and lowest for KneeOrdinalNet, which is the price of its better recovery of KL0 and KL2; a deployment aimed at fewer missed cases would have to shift the decision threshold using validation data.`),
  TCAP(`Table ${T.clin}`, "Screening view at the KL ≥ 2 threshold, test set, mean ± SD of three seeds. Missed: knees with true KL ≥ 2 predicted KL0 or KL1 (mean count of 721). Under/over ≥ 2: share of all test images graded two or more grades below/above the truth."),
  TBL([1900, 1500, 1500, 1200, 1500, 1472], [["Model", "Sensitivity", "Specificity", "Missed", "Under ≥ 2 (%)", "Over ≥ 2 (%)"], ...data.clin], { size: 17, shade: (r) => r[0] === "KneeOrdinalNet" }),
  GAP(),
  H2("4.3 ROC and precision–recall curves"),
  P(`Figure 9 shows one-versus-rest ROC and precision–recall curves per grade for the two models (seed 42). Both models separate the extreme grades (KL3, KL4) almost perfectly and KL0 well; the KL1 curves are clearly the lowest, and its precision–recall curve stays near 0.3–0.5, only modestly above the 18% prevalence of KL1 in the test set. The three-seed mean macro AUC values are 0.911 for KneeOrdinalNet and 0.899 for DeiT-Small (Table ${T.main}). For the seed-42 curves shown, the macro AUC is 0.906 versus 0.895, the micro AUC 0.928 versus 0.924, the macro precision–recall AUC 0.743 versus 0.733 and the micro precision–recall AUC 0.776 versus 0.750, so the ranking quality of the two models is close, with a small advantage for KneeOrdinalNet. The curves of the two models overlap for most grades, consistent with the small differences in Table ${T.main}.`),
  ...FIG("f_rocpr.png", 6.3, "Figure 9", "One-versus-rest ROC (left) and precision–recall (right) curves per grade, test set, seed 42. Solid: KneeOrdinalNet; dashed: DeiT-Small; thick black: macro average. KL1 is the hardest grade."),
  H2("4.4 Confidence intervals and calibration"),
  P(`Table ${T.ci} gives 95% percentile bootstrap intervals (1,000 resamples of the test set, seed 42) for KneeOrdinalNet, the hybrid, which was best on validation, and DeiT-Small. The intervals are about ±0.022 wide for macro F1 and overlap almost completely, so the test set alone does not resolve the ordering of the three models; the paired comparison is in Table ${T.pair}. Calibration follows.`),
  TCAP(`Table ${T.ci}`, "Test metrics with 95% bootstrap intervals (seed 42; 1,000 resamples, percentile method)."),
  TBL([2100, 2324, 2324, 2324], [["Metric", "KneeOrdinalNet", "Hybrid", "DeiT-Small"], ...data.ci], { size: 17 }),
  GAP(),
  P(`DeiT-Small is well calibrated (ECE 0.048); KneeOrdinalNet is not (ECE 0.129) and its reliability curve lies above the diagonal, i.e. it is under-confident (Figure 10). Temperature scaling fitted on validation (T below 1 for KneeOrdinalNet, sharpening the output) reduces its ECE to 0.084 and its Brier score from 0.435 to 0.411, but it remains worse than DeiT-Small (0.034 after scaling). The ablation without the ordinal loss has an ECE of 0.042 (Table ${T.cal}), which places the cause on component D: soft targets and label smoothing teach the network not to be certain.`),
  ...FIG("f_reliability.png", 5.8, "Figure 10", "Reliability diagrams, test set, seed 42 (15 bins, bins with at least 5 images). Temperature learned on validation."),
  TCAP(`Table ${T.cal}`, "Calibration on the test set before and after temperature scaling (mean of three seeds)."),
  TBL([2800, 1200, 1200, 1200, 1200, 1472], [["Model", "ECE", "ECE (temp.)", "Brier", "Brier (temp.)", "T"], ...data.cal], { size: 17 }),
  GAP(),
  H2("4.5 Ablations, control and statistical tests"),
  P(`Table ${T.abl} and Figure 11 give the ablations (three seeds each). On validation, removing the row attention caused the largest macro-F1 drop (−0.0106), so it was the target of the McNemar test; removing the other components changed validation macro F1 by less than 0.005, which is below the seed SD (about 0.01). On the test set removing A, B and C lowers macro F1 by 0.019, 0.007 and 0.010 and QWK by 0.013, 0.007 and 0.011, whereas removing D does not lower macro F1 (0.714) but lowers QWK (0.855) and increases MAE (0.340). Every single-component effect is within one to two seed SDs, so no component is decisive on its own. The components are cheap at inference: in a same-session CPU timing the full model needs 219.9 ms per image against 213.6 ms for the control (+2.9%), and each ablated variant lies between 215.8 and 220.2 ms; DeiT-Small needs 198.5 ms and ConvNeXt-Tiny 213.8 ms, so the ranking of the first session is preserved. The exact McNemar test between the full model and the version without row attention gave p = 0.097, 1.0 and 0.002 for seeds 42, 123 and 2026: significant for one seed only, so the contribution of row attention is not established.`),
  P(`The matched control is the key comparison. With the same tuned hyperparameters, the plain ConvNeXt-Tiny reaches a macro F1 of 0.707 ± 0.006, only 0.006 below KneeOrdinalNet (paired bootstrap significant in 0 of 3 seeds), and a QWK of 0.852 against 0.862 (significant in 2 of 3 seeds). Hence most of the macro-F1 improvement over the untuned ConvNeXt-Tiny (0.674) is explained by hyperparameter tuning, and the components add a small, more consistent improvement in ordinal agreement and distant errors (2.7% versus 2.9%). Table ${T.pair} summarizes the paired tests. KneeOrdinalNet is significantly better than ConvNeXt-Tiny on QWK and MAE in all three seeds, and than DeiT-Small on QWK and MAE in two of three.`),
  ...FIG("f_ablation.png", 6.3, "Figure 11", "Left: ablation of KneeOrdinalNet, test macro F1, three seeds (dots), parameter count at bar start. Right: macro F1 versus CPU latency."),
  TCAP(`Table ${T.abl}`, `Ablation results in the layout of the project description. Each row changes one component of KneeOrdinalNet and keeps all other settings fixed; mean ± SD of seeds 42, 123 and 2026. Design decisions were taken on validation; Δ is the difference of the mean validation macro F1 from the full model. CPU latency: median per image, batch 1, one thread, all rows timed in the same session; absolute values differ from Table ${T.main}, which was timed in an earlier session.`),
  TBL([850, 1750, 640, 1000, 1000, 640, 1000, 1000, 600], [["Experiment", "Variant", "Params (M)", "Val macro F1", "Val bal. acc.", "Δ val F1", "Test macro F1", "Test bal. acc.", "CPU (ms)"], ...data.abl2], { size: 14, shade: (r) => r[0] === "Full model" }),
  GAP(),
  TCAP(`Table ${T.pair}`, "Paired bootstrap differences on the test set: mean difference over seeds (number of seeds out of 3 with a 95% interval excluding 0). For MAE, negative values favour the first model."),
  TBL([3300, 1900, 1900, 1972], [["Comparison", "Macro F1", "QWK", "MAE"], ...data.pair], { size: 17 }),
  GAP(),
  H2("4.6 Hybrid model"),
  P(`On validation, the equal-weight mean of all four members was best (macro F1 0.716 ± 0.008) ahead of stacking (0.710), the ordinal decision rule (0.709) and weighted fusion (0.708) with the same four members (Table ${T.habl}). The ordinal rule did lower the distant-error rate (2.6% versus 4.7%), so it trades macro F1 for ordinal safety. Leave-one-out sets of three members scored 0.711–0.712 and pairs 0.697–0.709, so every member contributes a little. On test, the hybrid reaches 0.717 ± 0.003 macro F1 (best of all rows in Table ${T.main}), but it is not significantly better than KneeOrdinalNet in any seed (Table ${T.pair}); it is significantly better than DeiT-Small on QWK and MAE in all seeds. It costs 101.4 M parameters (387 MB) and 533 ms per image, about 3.6 and 3.7 times those of KneeOrdinalNet. For a gain of 0.004 macro F1, we do not consider this worthwhile.`),
  P(`Table ${T.habl} is the hybrid ablation: each member alone and each fusion method with all four members, on validation, because decisions rest on validation only. The best member alone, KneeOrdinalNet, reaches 0.703 ± 0.012, so the equal-weight mean adds 0.013 macro F1, which is comparable to the seed SD (0.006–0.012), for 3.6 times the parameters and latency. Stacking has the highest balanced accuracy (0.722) but the least stable macro F1 (SD 0.030), and the ordinal decision rule gives up 0.008 macro F1 for the fewest distant errors (2.6%).`),
  TCAP(`Table ${T.habl}`, "Hybrid ablation on validation, mean ± SD of three seeds. Δ: difference of the mean validation macro F1 from the selected equal-weight mean. CPU latency from the first timing session; fusion rows are the sum of the four members."),
  TBL([3200, 800, 1250, 1250, 800, 1000], [["Configuration", "Params (M)", "Val macro F1", "Val bal. acc.", "Δ val F1", "CPU (ms)"], ...data.habl], { size: 16, shade: (r) => r[0].startsWith("4 members, equal") }),
  GAP(),
  H2("4.7 Explainability"),
  P(`Figures 12 and 13 show Grad-CAM, rollout and Integrated Gradients for both models on the same test images (ten correct cases and six errors here; the other ten correct and six error cases are in Appendix B; labels give the true grade and each model's prediction). For KneeOrdinalNet, Grad-CAM forms horizontal bands at the joint line (peak row about 143 of 224) and at the lateral and medial margins, which is anatomically plausible: these are the places of joint-space narrowing and marginal osteophytes. DeiT-Small's Grad-CAM also follows the joint line but is more spread along it. Integrated Gradients is sharper and noisier and concentrates on bone edges. Attention rollout highlights mostly the image borders and is the least informative map.`),
  ...FIG("f_xai_ok.png", 4.9, "Figure 12", "Correctly classified test images: Grad-CAM, IG and rollout for both models (first ten of twenty)."),
  P(`Errors (Figure 13) are often shared by both models, for example three KL0 knees called KL2 by both (cases 315, 53 and 595). In these cases KneeOrdinalNet's Grad-CAM peaks at the lateral or medial image margin outside the joint space, which suggests the prediction there rests on peripheral cues, such as marginal bone contours or positioning, rather than on the joint line; this is an interpretation and was not verified with annotations. Case 1541 (true KL3, predicted KL1 by KneeOrdinalNet) is a distant error with a diffuse map.`),
  ...FIG("f_xai_bad.png", 5.4, "Figure 13", "Misclassified test images (distant errors first), same layout as Figure 12 (first six of twelve)."),
  P(`The deletion test (Figure 14) removes the most important pixels first and tracks the probability of the true class; the maps are computed for the true grade. A faster drop (lower area under the curve, AUC) means the map identifies pixels the model uses. On the 60 class-balanced images the mean true-class probability before deletion is 0.530 for KneeOrdinalNet and 0.611 for DeiT-Small. All attribution methods beat random deletion: for KneeOrdinalNet Grad-CAM 0.179, IG 0.243 and random 0.438; for DeiT-Small Grad-CAM 0.264, IG 0.247, rollout 0.317 and random 0.482. Grad-CAM is the most faithful method for KneeOrdinalNet, Grad-CAM and IG are similar for DeiT-Small, and rollout is the weakest, matching the visual impression. The Grad-CAM curve of KneeOrdinalNet falls below the value of a fully blurred image (about 0.20) after 30–60% deletion and then returns to it, so partly blurred images push the model towards other grades; the curve therefore measures more than lost evidence.`),
  ...FIG("f_deletion.png", 6.0, "Figure 14", "Deletion curves for KneeOrdinalNet (left) and DeiT-Small (right), 60 class-balanced test images, seed 42; the probability of the true class is tracked. Lower AUC is better; dotted: random order."),
  P(`**Attribution at the image border.** For a uniform map the border band holds a share of 0.203, and random maps give 0.203. KneeOrdinalNet puts 0.120 ± 0.055 of its Grad-CAM mass there (0.59 times the uniform share) and 0.084 ± 0.026 of its IG mass (0.41); DeiT-Small's Grad-CAM gives 0.051 ± 0.053 (0.25) and IG 0.157 ± 0.042 (0.78), while attention rollout gives 0.241 ± 0.011 (1.19), close to uniform and again uninformative. We therefore found no sign that either model relies on the outer frame of the image. The test is coarse: the joint is centred in this dataset, so a low border share is expected, markers or text inside the image are not covered, and the result rests on 60 images and one seed.`),
  P(`The randomization test (Figure 15) checks whether maps depend on the learned weights. For DeiT-Small the similarity to the original map drops to about zero as soon as the head is randomized; for KneeOrdinalNet it falls to 0.21 ± 0.38 after the head and about 0.02 after stage 3. Both models pass the sanity check, but KneeOrdinalNet's similarity stays between about 0.1 and 0.3 after early layers are randomized, so part of its map structure is driven by low-level image content and should be treated with care.`),
  ...FIG("f_randomization.png", 4.8, "Figure 15", "Model-parameter randomization test: Spearman similarity of Grad-CAM to the original map (mean ± SD of 20 images)."),
  H2("4.8 Challenging tasks"),
  P(`**Corruption robustness.** Figure 16 and Table ${T.corr} show macro F1 for KneeOrdinalNet, DeiT-Small and the control under four corruptions at five severities (seed 42; clean macro F1 0.704, 0.702 and 0.710). KneeOrdinalNet is almost immune to contrast loss (mean drop 0.003 versus 0.029 for DeiT-Small), which is exactly what per-image standardization (A) is designed for. It is, however, less robust than DeiT-Small to JPEG compression (drop 0.136 versus 0.095) and noise (0.188 versus 0.153), and similar for blur (0.104 versus 0.089). The control is the least robust to noise (drop 0.229), so components A–D help there but do not close the gap to the transformer, which seems to rely less on high-frequency detail.`),
  ...FIG("f_robust.png", 6.3, "Figure 16", "Test macro F1 under corruption at severities 1–5 (0 = clean), seed 42."),
  TCAP(`Table ${T.corr}`, "Mean test macro F1 over five severities (drop from the clean value), seed 42."),
  TBL([2400, 1668, 1668, 1668, 1668], [["Model", "Noise", "Blur", "JPEG", "Contrast loss"],
    ["KneeOrdinalNet", "0.516 (−0.188)", "0.600 (−0.104)", "0.568 (−0.136)", "0.701 (−0.003)"],
    ["DeiT-Small", "0.549 (−0.153)", "0.613 (−0.089)", "0.607 (−0.095)", "0.672 (−0.029)"],
    ["Control", "0.481 (−0.229)", "0.594 (−0.116)", "0.597 (−0.114)", "0.701 (−0.010)"]], { size: 17 }),
  GAP(),
  P(`**Selective prediction.** A model that can abstain on uncertain images is attractive for a screening aid. Figure 17 and Table ${T.sel} show risk–coverage curves with the maximum-probability confidence. The area under the risk–coverage curve (AURC, lower is better) is 0.143 for the hybrid, 0.147 for KneeOrdinalNet (0.146 after temperature scaling), 0.162 for the control and 0.177 for DeiT-Small. For ordinal agreement, KneeOrdinalNet benefits clearly from rejection: QWK rises from 0.856 to 0.905 at 79% coverage and to 0.954 at 52% coverage. Macro F1, however, hardly improves for KneeOrdinalNet (0.704 to about 0.72) while DeiT-Small rises to 0.758 at 51% coverage and the hybrid to 0.768. The confidence of the ordinal model is therefore informative about distant errors but not about minority-class errors, probably because soft labels flatten its output distribution near grade boundaries.`),
  ...FIG("f_selective.png", 6.3, "Figure 17", "Left: error rate against coverage (seed 42). Right: test macro F1 at coverages set on validation."),
  TCAP(`Table ${T.sel}`, "Selective prediction on the test set (seed 42). Coverage targets were set on validation."),
  TBL([2000, 1000, 1500, 1500, 1500, 1572], [["Model", "AURC", "QWK at ≈80% cov.", "F1 at ≈80% cov.", "QWK at ≈50% cov.", "F1 at ≈50% cov."],
    ["KneeOrdinalNet", "0.147", "0.905 (78.7%)", "0.724", "0.954 (51.7%)", "0.709"],
    ["DeiT-Small", "0.177", "0.884 (80.5%)", "0.718", "0.944 (50.9%)", "0.758"],
    ["Control", "0.162", "0.896 (79.2%)", "0.743", "0.939 (52.6%)", "0.765"],
    ["Hybrid", "0.143", "0.898 (81.0%)", "0.740", "0.956 (49.9%)", "0.768"]], { size: 16 }),
  GAP(),
];

const discussion = [
  H1("5. Discussion"),
  H2("5.1 Why the original model helped, and where it did not"),
  P(`KneeOrdinalNet is the best single model on every accuracy-type metric and, most consistently, on ordinal metrics: QWK 0.862, MAE 0.332 and 2.7% distant errors, against 0.840, 0.363 and 3.8% for DeiT-Small. The evidence for why is mixed. The ordinal loss (D) is the component with the most direct effect on the quantities it targets: removing it raises MAE and lowers QWK while leaving macro F1 unchanged. Per-image standardization (A) explains the near-immunity to contrast loss, and removing it gave the lowest test macro F1 of the ablations (0.694). Multi-scale pooling (B) and row attention (C) each cost 0.007–0.010 macro F1 when removed, but these differences are comparable to the seed-to-seed SD and the McNemar test was significant for one seed only. The control experiment gives the sober summary: the plain ConvNeXt-Tiny with the same tuned hyperparameters reaches 0.707, so about 85% of the improvement over the untuned ConvNeXt-Tiny is tuning, and the custom components add roughly 0.006 macro F1 (not significant) and 0.010 QWK (significant in two of three seeds). The design is a modest, honest gain rather than a breakthrough.`),
  P(`Some results did not meet expectations. The design log shows that the full architecture did not raise validation macro F1 before tuning; KL1 recall fell from 0.53 (control) to 0.43; calibration worsened markedly because of the ordinal loss; robustness to noise and JPEG compression was lower than for DeiT-Small; and the confidence of the ordinal model is a poor selector for minority-class errors. KL1 is the clearest failure: soft labels deliberately spread probability to neighbouring grades, and KL1 lies between the two largest classes (KL0, KL2), so a model trained this way can drift toward KL0 or KL2 on doubtful cases. Because KL1 itself is where human readers disagree most [19,20], part of this error is label noise that no model can remove.`),
  H2("5.2 Comparison with recent studies"),
  P(`Direct comparison is limited by different protocols (Table ${T.lit}). On the same kneeKL224 test set, VL-OrdinalFormer [14] reports an accuracy of 70.3% and a macro F1 of 70.2% with a ViT-L/16 backbone (hundreds of millions of parameters, five-fold ensemble, CLIP distillation and test-time augmentation). KneeOrdinalNet reaches 69.4% and 0.713 with a single 28 M-parameter network without test-time augmentation; the two are statistically indistinguishable at this test size, at a fraction of the cost, though [14] does not report QWK or MAE. Studies using the same 8,260 images with ordinal heads report a QWK of 0.830 and macro F1 of 0.676 (VGG-16 with CORAL [15]) and a QWK of 0.839 with macro F1 0.675 (ResNet-18 ordinal regression [16]); our QWK of 0.862 and macro F1 of 0.713 are higher, although those authors used smaller or older networks and probably less tuning, so part of the gap is expected. The six-model ensemble of Pi et al. [7] reports 76.9% accuracy and a macro F1 of 0.767, clearly higher than ours; however, their protocol is described as pooled cross-validation over training and validation data and it is unclear whether the figure refers to the official test set, so we cannot confirm that it is comparable. Reported accuracies of 94–98% [8,10,12] and the 78.4% of [13] use merged or unspecified data and splits; with an unknown split and without ordinal or per-class metrics they cannot be compared with our results. Consistent with [7] and [15], KL1 was the weakest class in our study (F1 0.39–0.42).`),
  H2("5.3 Error types, strengths and disadvantages"),
  P(`**Error types.** Errors are mostly adjacent: only 2.7% of test images are off by two or more grades. The most common confusions are KL1 with KL0 and KL2, and KL2 with KL1. Several misclassifications are shared across models, which suggests ambiguous images or noisy labels rather than architecture-specific failures. **Strengths.** The model has the best ordinal agreement among single models, the lowest distant-error rate, strong robustness to contrast changes, only 2.1% more parameters than its backbone, a CPU latency only about 3% above the plain ConvNeXt-Tiny (220 versus 214 ms in the same session), and anatomically plausible Grad-CAM maps that pass the randomization and deletion tests. **Disadvantages.** Its macro-F1 advantage is small and not clearly significant; it is the worst-calibrated model; it has lower KL1 recall; it is less robust to noise and compression than the transformer; and its attention modules are not clearly justified by their ablation results. The hybrid gains almost nothing for 3.6 times the parameters.`),
  H2("5.4 Generalizability and limitations"),
  BUL(`**Not ready for clinical use.** The model is a research prototype. It must not be used for diagnosis or treatment decisions.`),
  BUL(`**No external validation.** All images come from one source collection (processed Osteoarthritis Initiative radiographs). The model has not been tested on other hospitals, X-ray devices, acquisition protocols, post-operative knees, implants, or populations outside the original cohort; the dataset does not provide the demographic or device metadata needed to examine subgroups.`),
  BUL(`**Image preprocessing.** The 224 × 224 cropped images are lower resolution than clinical radiographs, and the knees are pre-localized, so a real system would need a detection step.`),
  BUL(`**Label noise.** KL grades are subjective [19,20]. No second reading was available, so performance may be capped by label noise, especially for KL1.`),
  BUL(`**Selection effects.** Four models were tuned and a hybrid was chosen among 44 configurations on a validation set of 826 images (27 KL4 images), so the validation estimates are optimistic; test numbers come from a single evaluation, but the test set is one fixed sample, and differences below about 0.02 macro F1 are within bootstrap noise.`),
  BUL(`**Compute and seeds.** Hyperparameter search used a 40% training subset and a reduced epoch budget; most single-seed models (nine of twelve networks) were not repeated; CPU latency was measured once per model (seed 42) in two separate sessions with different absolute values; DeiT-Small stopped at its epoch budget rather than by early stopping.`),
  BUL(`**Challenge tasks.** Corruption and selective-prediction results use seed 42 only; a data-fraction learning curve was not done. The reading of peripheral cues in error cases (Section 4.7) has not been checked with expert annotations.`),
];

const conclusion = [
  H1("6. Conclusion"),
  P(`Under a leakage-free, validation-only protocol on kneeKL224, a ConvNeXt-Tiny network with four dataset-motivated components (KneeOrdinalNet) is the best single model, with a macro F1 of 0.713 ± 0.008, a QWK of 0.862 and a mean absolute grade error of 0.332, compared with 0.698, 0.840 and 0.363 for the best existing model, DeiT-Small. The answer to the research question is therefore a cautious yes for ordinal agreement and distant errors and a weak yes for macro F1: against a matched control with identical hyperparameters the macro-F1 gain is 0.006 and not significant, so most of the improvement is due to hyperparameter tuning. The price is worse calibration, lower KL1 recall and lower robustness to noise and compression than DeiT-Small. A four-model ensemble reached 0.717 but at 3.6 times the size. The explanations pass randomization and deletion checks, and the model's confidence is useful for rejecting distant errors. The model is not ready for clinical use.`),
  P(`Feasible next steps are an ordinal head with a calibration-aware loss (for example, CORAL combined with temperature-aware training), a two-stage design that first separates KL ≤ 2 from KL ≥ 3 where reader agreement is higher, testing on an independent external cohort, and annotation-based validation of the saliency maps.`),
];

const refs = [
  `Kellgren JH, Lawrence JS. Radiological assessment of osteo-arthrosis. Ann Rheum Dis. 1957;16(4):494–502. doi:10.1136/ard.16.4.494`,
  `Antony J, McGuinness K, O'Connor NE, Moran K. Quantifying radiographic knee osteoarthritis severity using deep convolutional neural networks. Proc. International Conference on Pattern Recognition (ICPR); 2016. arXiv:1609.02469`,
  `Tiulpin A, Thevenot J, Rahtu E, Lehenkari P, Saarakkala S. Automatic knee osteoarthritis diagnosis from plain radiographs: a deep learning-based approach. Sci Rep. 2018;8:1727. doi:10.1038/s41598-018-20132-7`,
  `Chen P, Gao L, Shi X, Allen K, Yang L. Fully automatic knee osteoarthritis severity grading using deep neural networks with a novel ordinal loss. Comput Med Imaging Graph. 2019;75:84–92. doi:10.1016/j.compmedimag.2019.06.002`,
  `Thomas KA, Kidziński Ł, Halilaj E, Fleming SL, Venkataraman GR, Oei EHG, Gold GE, Delp SL. Automated classification of radiographic knee osteoarthritis severity using deep neural networks. Radiol Artif Intell. 2020;2(2):e190065. doi:10.1148/ryai.2020190065`,
  `Yong CW, Teo K, Murphy BP, Hum YC, Tee YK, Xia K, Lai KW. Knee osteoarthritis severity classification with ordinal regression module. Multimed Tools Appl. 2022;81(29):41497–41509. doi:10.1007/s11042-021-10557-0`,
  `Pi S-W, Lee B-D, Lee MS, Lee HJ. Ensemble deep-learning networks for automated osteoarthritis grading in knee X-ray images. Sci Rep. 2023;13:22887. doi:10.1038/s41598-023-50210-4`,
  `Tariq T, Suhail Z, Nawaz Z. Knee osteoarthritis detection and classification using X-rays. IEEE Access. 2023;11:48292–48303. doi:10.1109/ACCESS.2023.3276810`,
  `Yoon JS, Yon C-J, Lee D, Lee JJ, Kang CH, Kang S-B, Lee N-K, Chang CB. Assessment of a novel deep learning-based software developed for automatic feature extraction and grading of radiographic knee osteoarthritis. BMC Musculoskelet Disord. 2023;24:869. doi:10.1186/s12891-023-06951-4`,
  `Jahan M, Hasan MZ, Samia IJ, Fatema K, Rony MAH, Arefin MS, Moustafa A. KOA-CCTNet: an enhanced knee osteoarthritis grade assessment framework using modified compact convolutional transformer model. IEEE Access. 2024;12:107719–107741. doi:10.1109/ACCESS.2024.3435572`,
  `Lee DW, Song DS, Han H-S, Ro DH. Accurate, automated classification of radiographic knee osteoarthritis severity using a novel method of deep learning: plug-in modules. Knee Surg Relat Res. 2024;36:24. doi:10.1186/s43019-024-00228-3`,
  `Kılıç Ş. DenseNet201-based knee osteoarthritis grading with multi-scale attention. PeerJ Comput Sci. 2025;11. doi:10.7717/peerj-cs.3329`,
  `Qiu R, Rahim SKNA, Jamil N, Hamzah R. DualVitOA: dual vision transformer for knee osteoarthritis grading. IAES Int J Artif Intell. 2025;14(2):925–932. doi:10.11591/ijai.v14.i2.pp925-932`,
  `Ullah Z, Kim J. VL-OrdinalFormer: vision–language guided ordinal transformers for knee osteoarthritis grading. Mathematics. 2026;14(6):963; preprint arXiv:2601.00879`,
  `Kizilkaya H, Ortataş FN, Üreten K. Knee osteoarthritis severity classification with ordinal regression (VGG-16 and CORAL). Bozok Med J. 2026;16(1):115–125. doi:10.16919/bozoktip.1859321`,
  `Khatami MK, Indriani F, Farmadi A, Muliadi, Mazdadi MI. Evaluating ordinal regression approaches for automated knee osteoarthritis severity classification using ResNet-18 on radiographic X-ray images. J Vocational Inform Comput Educ. 2026;4(2). doi:10.66053/voice.v4i2.845`,
  `Li X, Zhou R, Yan X, Yan L, Li Z, Zhu C, Fu R, Chai Y. AGE-Net: spectral–spatial fusion and anatomical graph reasoning with evidential ordinal regression for knee osteoarthritis grading. arXiv:2601.17336 (preprint, not peer reviewed); 2026`,
  `Wang Y, Wang X, Gao T, Du L, Liu W. An automatic knee osteoarthritis diagnosis method based on deep learning: data from the Osteoarthritis Initiative. J Healthc Eng. 2021;2021:5586529. doi:10.1155/2021/5586529`,
  `Eijking HM, Voskuilen R, Tilman P, Schotanus MGM, Jong B, Boonen B, Most J. Reliability of radiographic knee and hip osteoarthritis classification between radiologists and orthopaedic surgeons. J Orthop Surg Res. 2026;21:174. doi:10.1186/s13018-026-06695-6`,
  `Beyaz S, Yayli SB, Kılıç K. From variability to consistency: building a Kellgren–Lawrence gonarthrosis dataset. J Orthop Surg Res. 2025;20:922. doi:10.1186/s13018-025-06057-8`,
  `Liu Z, Mao H, Wu C-Y, Feichtenhofer C, Darrell T, Xie S. A ConvNet for the 2020s. Proc. IEEE/CVF CVPR; 2022:11976–11986`,
  `Touvron H, Cord M, Douze M, Massa F, Sablayrolles A, Jégou H. Training data-efficient image transformers and distillation through attention. Proc. ICML; PMLR 139:10347–10357; 2021`,
  `Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. Proc. ICML; PMLR 70:1321–1330; 2017`,
  `Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. Grad-CAM: visual explanations from deep networks via gradient-based localization. Proc. ICCV; 2017. doi:10.1109/ICCV.2017.74`,
  `Sundararajan M, Taly A, Yan Q. Axiomatic attribution for deep networks. Proc. ICML; PMLR 70:3319–3328; 2017`,
  `Adebayo J, Gilmer J, Muelly M, Goodfellow I, Hardt M, Kim B. Sanity checks for saliency maps. Advances in Neural Information Processing Systems 31; 2018:9505–9515`,
  `Akiba T, Sano S, Yanase T, Ohta T, Koyama M. Optuna: a next-generation hyperparameter optimization framework. Proc. KDD; 2019:2623–2631. doi:10.1145/3292500.3330701`,
  `Cao W, Mirjalili V, Raschka S. Rank consistent ordinal regression for neural networks with application to age estimation. Pattern Recognit Lett. 2020;140:325–331. doi:10.1016/j.patrec.2020.11.008`,
  `Chen P. Knee Osteoarthritis Severity Grading Dataset. Mendeley Data, V1. doi:10.17632/56rmx5bjcr.1 (CC BY 4.0). https://data.mendeley.com/datasets/56rmx5bjcr/1`,
  `Hu J, Shen L, Sun G. Squeeze-and-excitation networks. Proc. IEEE/CVF CVPR; 2018:7132–7141`,
  `Woo S, Park J, Lee J-Y, Kweon IS. CBAM: convolutional block attention module. Proc. ECCV; 2018:3–19. doi:10.1007/978-3-030-01234-2_1`,
  `Díaz R, Marathe A. Soft labels for ordinal regression. Proc. IEEE/CVF CVPR; 2019:4738–4747. doi:10.1109/CVPR.2019.00487`,
  `Lin T-Y, Dollár P, Girshick R, He K, Hariharan B, Belongie S. Feature pyramid networks for object detection. Proc. IEEE CVPR; 2017:2117–2125`,
  `Paszke A, Gross S, Massa F, Lerer A, Bradbury J, Chanan G, Killeen T, Lin Z, Gimelshein N, Antiga L, Desmaison A, Köpf A, Yang E, DeVito Z, Raison M, Tejani A, Chilamkurthy S, Steiner B, Fang L, Bai J, Chintala S. PyTorch: an imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems 32; 2019:8024–8035`,
  `Wightman R. PyTorch Image Models (timm). GitHub repository; 2019. doi:10.5281/zenodo.4414861`,
  `Kokhlikyan N, Miglani V, Martin M, Wang E, Alsallakh B, Reynolds J, Melnikov A, Kliushkina N, Araya C, Yan S, Reblitz-Richardson O. Captum: a unified and generic model interpretability library for PyTorch. arXiv:2009.07896; 2020`,
  `Gildenblat J, contributors. PyTorch library for CAM methods (pytorch-grad-cam). GitHub repository; 2021. https://github.com/jacobgil/pytorch-grad-cam`,
  `Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, Blondel M, Prettenhofer P, Weiss R, Dubourg V, Vanderplas J, Passos A, Cournapeau D, Brucher M, Perrot M, Duchesnay É. Scikit-learn: machine learning in Python. J Mach Learn Res. 2011;12:2825–2830`,
];
const refParas = refs.map((r, i) => new Paragraph({ children: [new TextRun({ text: `[${i + 1}] `, font: FONT, size: 19, bold: true }), ...runs(r, { size: 19 })], spacing: { line: 240, after: 50 }, indent: { left: 480, hanging: 480 }, alignment: AlignmentType.LEFT }));

const appendix = [
  H1("Appendix A. Best hyperparameters"),
  P(`Table ${T.hp} lists the tuned values that were retrained on the full training set. The control uses the same values as KneeOrdinalNet. The learning rates are for the head and the fine-tuned backbone; all networks used AdamW, cosine decay, label smoothing as listed and a weighted sampler. Default (untuned) values were learning rates 1e-3 / 1e-4, weight decay 0.01, label smoothing 0.1, dropout 0.3 and batch size 64. The HOG+SVM used C = 0.01 with class weights.`),
  TCAP(`Table ${T.hp}`, "Best hyperparameters found by Optuna."),
  TBL([1900, 700, 1600, 1000, 1000, 800, 1000, 1072], [["Model", "Batch", "LR head / backbone", "Weight decay", "Label smoothing", "Dropout", "Unfrozen fraction", "σ / λ (ordinal)"], ...data.hp], { size: 16 }),
  GAP(),
  H1("Appendix B. Additional explanation examples"),
  P(`The remaining ten correctly classified images (Figure B1) and six misclassified images (Figure B2) complete the sets of twenty correct and twelve incorrect cases.`),
  ...FIG("f_xai_ok_b.png", 4.9, "Figure B1", "Correctly classified test images, second ten of twenty."),
  ...FIG("f_xai_bad_b.png", 5.4, "Figure B2", "Misclassified test images, second six of twelve."),
  H1("Appendix C. Experiment log summary"),
  P(`Table ${T.tt} summarizes training runs of seed 42 (epochs run, best epoch, training time on the T4 GPU, validation macro F1). The summed logged training time over all 48 logged runs, including repeated seeds, was ${data.total_h.toFixed(1)} hours.`),
  TCAP(`Table ${T.tt}`, "Training log, seed 42."),
  TBL([3300, 1500, 1500, 1500, 1272], [["Model", "Epochs run", "Best epoch", "Time (min)", "Val macro F1"], ...data.tt], { size: 17 }),
  GAP(),
  H1("Appendix D. Per-network results"),
  P(`This appendix gives, for every neural network, the per-class precision, sensitivity (recall), specificity and F1 on the test set (Tables ${T.apxp}–${T.apxf}; mean of three seeds for models marked *, seed 42 otherwise), the confusion matrices (Figure A1) and further metrics (Table ${T.apxe}). Table ${T.ablx} adds the remaining test metrics of the ablation variants.`),
  TCAP(`Table ${T.ablx}`, "Further test metrics of the ablation variants (mean ± SD of three seeds)."),
  TBL([1900, 760, 600, 1050, 1050, 950, 950, 1000, 822], [["Variant", "Params (M)", "Val F1", "Test macro F1", "Test QWK", "Test MAE", "Distant err.", "ECE", "CPU lat. (ms)"], ...data.abl], { size: 16, shade: (r) => r[0] === "Full model" }),
  GAP(),
  TCAP(`Table ${T.apxp}`, "Per-class precision, test set."),
  TBL([2400, 1334, 1334, 1334, 1335, 1335], [["Model", "KL0", "KL1", "KL2", "KL3", "KL4"], ...data.apx.prec], { size: 17 }),
  GAP(),
  TCAP(`Table ${T.apxs}`, "Per-class sensitivity (recall), test set."),
  TBL([2400, 1334, 1334, 1334, 1335, 1335], [["Model", "KL0", "KL1", "KL2", "KL3", "KL4"], ...data.apx.sens], { size: 17 }),
  GAP(),
  TCAP(`Table ${T.apxk}`, "Per-class specificity, test set."),
  TBL([2400, 1334, 1334, 1334, 1335, 1335], [["Model", "KL0", "KL1", "KL2", "KL3", "KL4"], ...data.apx.spec], { size: 17 }),
  GAP(),
  TCAP(`Table ${T.apxf}`, "Per-class F1, test set."),
  TBL([2400, 1334, 1334, 1334, 1335, 1335], [["Model", "KL0", "KL1", "KL2", "KL3", "KL4"], ...data.apx.f1], { size: 17 }),
  GAP(),
  TCAP(`Table ${T.apxe}`, "Further test metrics: weighted F1, QWK, MAE (grades), adjacent-error rate, distant-error rate and macro precision–recall AUC."),
  TBL([2200, 1100, 1100, 1100, 1200, 1200, 1172], [["Model", "Weighted F1", "QWK", "MAE", "Adjacent err.", "Distant err.", "PR AUC"], ...data.extra], { size: 17 }),
  GAP(),
  ...FIG("f_conf_all.png", 6.3, "Figure A1", "Row-normalized test confusion matrices of all neural networks, seed 42 (true grade in rows)."),
  P(`Figures A2 and A3 show the training curves of the twelve networks of the main comparison (seed 42; tuned settings where a search was run). The thin vertical line marks the change from the head-training to the fine-tuning stage for pretrained networks, and the dotted line marks the epoch with the best validation macro F1 that was kept.`),
  ...FIG("f_app_loss.png", 6.0, "Figure A2", "Training and validation loss per epoch of the twelve networks, seed 42; dotted line: selected epoch."),
  ...FIG("f_app_f1.png", 6.0, "Figure A3", "Validation macro F1 per epoch of the twelve networks, seed 42; dotted line: selected epoch."),
  H1("Appendix E. Generative AI use statement"),
  P(`Generative AI (the assistant Claude, from Anthropic) was used throughout this project as a coding and writing assistant: to write and debug the Python code for the data audit, training, hyperparameter search, evaluation, statistics, explainability and figures; to prepare the Colab notebooks; to compute and format the results tables from the experiment outputs; and to draft and edit the English text of this report. The experiments were executed by the student on Google Colab, and all numbers in this report come from these runs. The student reviewed the code, results and text and is responsible for all content. The literature was located with web search and the bibliographic details of cited works were checked against publisher or repository pages; studies for which results could not be verified are cited without numbers.`),
];

// ================= DOCUMENT =================
const pageP = { size: { width: 11906, height: 16838 }, margin: { top: 1417, bottom: 1417, left: 1417, right: 1417 } };
const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 })] })] });
const doc = new Document({
  creator: "Baha Türk", title: "Grading Knee Osteoarthritis Severity with a Dataset-Specific Ordinal Network",
  styles: {
    default: { document: { run: { font: FONT, size: BODY } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 28, bold: true }, paragraph: { spacing: { before: 280, after: 120 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 24, bold: true }, paragraph: { spacing: { before: 200, after: 100 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 22, bold: true, italics: true }, paragraph: { spacing: { before: 160, after: 80 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
  ] },
  sections: [
    { properties: { page: pageP }, children: cover },
    { properties: { page: pageP }, footers: { default: footer }, children: [...abstract, ...intro, ...lit, ...methods, ...resultsMain] },
    { properties: { page: { size: { width: 11906, height: 16838, orientation: PageOrientation.LANDSCAPE }, margin: pageP.margin } }, footers: { default: footer }, children: mainTable },
    { properties: { page: pageP }, footers: { default: footer }, children: [...resultsRest, ...discussion, ...conclusion] },
    { properties: { page: pageP }, footers: { default: footer }, children: [H1("References"), ...refParas, ...appendix] },
  ],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync("report.docx", b); console.log("ok", b.length); });
