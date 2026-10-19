from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

d=Document()
s=d.sections[0]
s.left_margin=s.right_margin=Inches(0.6); s.top_margin=Inches(0.55); s.bottom_margin=Inches(0.55)
st=d.styles['Normal']; st.font.name='Calibri'; st.font.size=Pt(10)
st.paragraph_format.space_after=Pt(3); st.paragraph_format.space_before=Pt(0)

def H(t,size=11.5,before=7):
    p=d.add_paragraph(); p.paragraph_format.space_before=Pt(before); p.paragraph_format.space_after=Pt(2)
    r=p.add_run(t); r.bold=True; r.font.size=Pt(size); r.font.color.rgb=RGBColor(0x1F,0x3A,0x5F)
    p.paragraph_format.keep_with_next=True
    return p
def P(t,lead=None,italic=False,size=None):
    p=d.add_paragraph()
    if lead:
        r=p.add_run(lead); r.bold=True
        if size: r.font.size=Pt(size)
    r=p.add_run(t); r.italic=italic
    if size: r.font.size=Pt(size)
    return p
def B(t,lead=None,size=None):
    p=d.add_paragraph(style='List Bullet'); p.paragraph_format.space_after=Pt(1.5)
    if lead:
        r=p.add_run(lead); r.bold=True
        if size: r.font.size=Pt(size)
    r=p.add_run(t)
    if size: r.font.size=Pt(size)
def shade(c,color):
    tcPr=c._tc.get_or_add_tcPr(); sh=OxmlElement('w:shd'); sh.set(qn('w:val'),'clear'); sh.set(qn('w:fill'),color); tcPr.append(sh)
def no_split(row):
    trPr=row._tr.get_or_add_trPr(); e=OxmlElement('w:cantSplit'); trPr.append(e)
def repeat_header(row):
    trPr=row._tr.get_or_add_trPr(); e=OxmlElement('w:tblHeader'); trPr.append(e)
def table(hdr,rows,widths,fs=8,center_from=None,section_rows=True,keep=False):
    tb=d.add_table(rows=1,cols=len(hdr)); tb.style='Table Grid'; tb.autofit=False
    tb.alignment=WD_TABLE_ALIGNMENT.CENTER
    for i,h in enumerate(hdr):
        c=tb.rows[0].cells[i]; c.text=''; r=c.paragraphs[0].add_run(h); r.bold=True; r.font.size=Pt(fs); shade(c,'D9E2F3')
    repeat_header(tb.rows[0]); no_split(tb.rows[0])
    for row in rows:
        rw=tb.add_row(); no_split(rw); cells=rw.cells
        sec = section_rows and all(x=='' for x in row[1:])
        for i in range(len(hdr)):
            cells[i].text=''
            r=cells[i].paragraphs[0].add_run(row[i]); r.font.size=Pt(fs)
            if sec: r.bold=True; shade(cells[i],'F2F2F2')
            if center_from is not None and i>=center_from: cells[i].paragraphs[0].alignment=WD_ALIGN_PARAGRAPH.CENTER
        if sec:
            m=cells[0].merge(cells[len(hdr)-1])
            for p_ in list(m.paragraphs)[1:]:
                if not p_.text.strip(): p_._element.getparent().remove(p_._element)
    for r_ in tb.rows:
        for i,c in enumerate(r_.cells):
            for p in c.paragraphs: p.paragraph_format.space_after=Pt(0)
    if keep:
        for r_ in tb.rows[:-1]:
            for c in r_.cells:
                for p in c.paragraphs: p.paragraph_format.keep_with_next=True
    tblPr=tb._tbl.tblPr; lay=OxmlElement('w:tblLayout'); lay.set(qn('w:type'),'fixed'); tblPr.append(lay)
    grid=tb._tbl.tblGrid
    for i,gc in enumerate(grid.findall(qn('w:gridCol'))):
        gc.set(qn('w:w'),str(int(widths[i]*1440)))
    for r_ in tb.rows:
        for i,w in enumerate(widths):
            try: r_.cells[i].width=Inches(w)
            except Exception: pass
    return tb

# ------------------------------------------------------------------ title
t=d.add_paragraph(); r=t.add_run('Tox21 adult baseline and preliminary pediatric drug-safety transfer: full research update'); r.bold=True; r.font.size=Pt(15)
p=d.add_paragraph(); r=p.add_run('Pranav Mamidala  |  September 30, 2026  |  Preliminary results, prepared for PI discussion  |  Revised: now covers the two Colab notebooks and all experiments run on the box'); r.italic=True; r.font.size=Pt(9)

# ------------------------------------------------------------------ 1
H('1. Research question and scope correction')
P('The question, as it appears in the notebooks: does adding biological bioactivity signatures to chemical structure improve toxicity prediction, and could such a benefit carry from adult-oriented data to pediatric drug safety? Later box experiments asked the same transfer question on pediatric adverse-event data, then asked related questions about uncertainty, abstention and age-dependent biology.')
P('This work concerns small-molecule compounds and preliminary drug-safety transfer. It is not about metal-organic frameworks (MOFs). No notebook and no box experiment contains MOF data, code or results (Section 7). The adult data in the notebooks are 12 Tox21 in-vitro assays (nuclear-receptor and stress-response pathway activity). They are not adult clinical outcomes. The pediatric data are of three different kinds (adverse-event reports from FDA FAERS in the notebooks; adverse-event incidence from ClinicalTrials.gov results; and FAERS drug-event reporting signals in the later experiments). Each is a different endpoint from Tox21 assay activity and from each other. Every result here is preliminary and exploratory.','Scope correction. ')
P('Metrics are never pooled across the three families: Tox21 AUROC/AUPRC (Part I); pediatric Spearman correlation, interval width and coverage on CT.gov incidence (E1 to E8); and FAERS reporting-signal Spearman and MAE (E9 to E12). Numbers in Part II are quoted as printed in the named artifact files. Where I could not verify something, I say so.','How to read the numbers. ',italic=False)

# ------------------------------------------------------------------ 2
H('2. Overview of all experiments')
hdr=['ID','Experiment','Question','Data / n','Headline result (own metric)','Verdict (from artifact)']
W=[0.4,1.15,1.5,1.45,1.6,1.2]
rows=[
('Part I: Colab notebooks (Tox21, adult in-vitro assays; AUROC/AUPRC)','','','','',''),
('NB1','Adult baselines + early FAERS probe','Which model/feature family is best under scaffold CV?','Tox21: 7,587 compounds, 12 tasks; FAERS probe 137 compounds','Pooled AUROC 0.825, AUPRC 0.430 (RF Morgan+desc); probe AUROC 0.44 to 0.54, MCC 0','Baseline established; pediatric probe uninformative'),
('NB2','Signatures vs chemistry, with permuted controls','Do Chemical Checker signatures add to chemistry?','Same Tox21 set; probe 46 compounds (4 negative)','XGB pooled AUROC 0.8224 (chem) vs 0.8254 (+true signatures)','Null (no test run)'),
('Part II: box experiments (pediatric endpoints; not comparable to Part I)','','','','',''),
('E1','Aligned transfer v2 (preliminary)','Does ChEMBL target profile + chem beat chem for adult→pediatric CT.gov AE incidence?','434 adult / 927 ped structures; 15 endpoints','Mean ped Spearman: chem 0.171, +bio 0.174, permuted 0.172','REJECTS multimodal benefit'),
('E2','Aligned transfer v3 (final)','Same, with full adult-priority mapping','2,856 adult / 835 ped structures (634 overlap); 23,602 / 6,734 labels','chem 0.1244, +bio 0.1296, permuted 0.1222, bio-only 0.0850','REJECTS (unchanged vs E1)'),
('E3','V3 independent review','Is the E2 null sound and reproducible?','Read-only audit of E2','Metrics reproduced (0 mismatches); only BH-significant contrast is negative (−0.091)','REJECTS preserved; 3 critical issues'),
('E4','Ontogeny V4 (preliminary)','Do age-dependent enzyme features beat static chem?','V3 data; 13 enzymes/transporters; 15 endpoints','chem 0.1240 vs chem+ontogeny 0.1157','NO_SUPPORTED_​BENEFIT; Phase 2 not run'),
('E5','PACT','Can a small pediatric calibration set give valid, narrow intervals?','167 / 167 / 501 structures (dev/cal/test); 4,035 test rows','Coverage 0.902 (global 0.923); width 5.25 vs 6.08 (logit)','SUPPORTED (interval criteria only)'),
('E6','PACT validation v2','Does PACT beat the strongest fair baseline?','Same test set (reused, exploratory)','Width 5.251 vs 5.388 (B7); repeated splits 4.974 vs 4.998','Reproduction passed; advantage inconclusive'),
('E7','REACT-Fuse','Does reliability-gated biology help beyond chem?','835 ped structures; 474 (56.8%) with usable biology','Residual gate: real −0.048 vs shuffled −0.045','NOT_SUPPORTED (STOP/GO failed)'),
('E8','PACT-AL','Does conformal active learning beat random acquisition?','5-fold CV; 50 runs; 808 ped structures','ΔAULC vs random +0.0056 [−0.0025, +0.0133]','PARTIALLY_​SUPPORTED'),
('E9','DART (FAERS)','Do continuous-age trajectories beat age-group baselines?','FAERS 2018 to 2026Q2; test 103,940 cells','wSpearman 0.531 vs best baseline 0.664','PARTIALLY_​SUPPORTED (4/9 criteria); worse than best baseline'),
('E10','CCAD (FAERS)','Do conformal alarms / selective forecasts work?','980,709 quarter-level signal rows','Selective ΔMAE 0.018 (needed 0.02); monitoring unevaluable','NOT_SUPPORTED (pilot stop; final test unopened)'),
('E11','SCSP (FAERS)','Does stage-constrained abstention beat fair baselines?','Test 103,940 cells; 297 drugs × 298 events','ΔMAE 0.0000 vs strongest baseline','NOT_SUPPORTED (dev gate failed)'),
('E12','Adaptive borrowing (FAERS)','Can a prediction-time rule tell when adult history helps?','Dev sparse subset n = 1,666','Adult gain 0.0139 (needed ≥ 0.02)','NOT_SUPPORTED; stopped by headroom gate'),
('E13','SAFE-Ped','Frozen B4 + age-fair conformal abstention','No artifacts found on the box','Not verifiable','Not reported (see Section 8)'),
('Adjacent (not part of the transfer question)','','','','',''),
('A1','ToxLens baseline reproduction','Do published ECFP4 baselines reproduce?','External 11-endpoint benchmark (frozen author splits)','RF macro test AUROC 0.7806 (published 0.7796)','Reproduced; extension not run'),
]
table(hdr,rows,W,fs=7.5)
P('FAERS = FDA Adverse Event Reporting System; CT.gov = ClinicalTrials.gov; BH = Benjamini-Hochberg; AULC = area under the learning curve; wSpearman = reliability-weighted macro Spearman.',italic=True,size=8)

# ------------------------------------------------------------------ 3
H('3. Part I: the two Colab notebooks (adult Tox21 and an early pediatric probe)')
H('Methods (compact)',10.5,3)
B(' DeepChem Tox21 (7,831 rows, 12 tasks). RDKit cleanup, largest-fragment (salt removal) and neutralization; 8 unparseable SMILES dropped; 234 duplicates collapsed (per-task label mean; an exact 0.5 conflict becomes missing); compounds with no remaining label dropped; 2 round-trip failures dropped. Final set: 7,587 compounds, 75,521 labels.','Curation.')
B(' Morgan fingerprint (radius 2, 2,048 bits); 214 RDKit descriptors (log-scaled Ipc, columns with ≥5% missing dropped, median imputation, constant columns removed). Notebook 2 adds Chemical Checker signatures (spaces B1, B2, B4, B5, C3, C4, C5, D1, D2; 128 dimensions each; 1,152 total). Structural (A) and clinical-outcome (E) spaces were excluded on purpose to limit leakage.','Features.')
B(' A single Murcko-scaffold 80/10/10 split was tried first. It was unstable: some test tasks had 18 to 22 positives, and for NR-AR and SR-MMP no test-positive scaffold appeared in training. I then moved to 5-fold scaffold-grouped CV (fold sizes 1698/1473/1472/1472/1472).','Splits.')
B(' Random forest (500 trees, balanced class weights), XGBoost (400 trees, depth 6, learning rate 0.05, positive-class weighting) and a weighted PyTorch MLP (512-128, dropout 0.3, positive-class weights). One model per task.','Models.')
B(' Missing labels are masked in training and scoring, never imputed. A task/fold is scored only if the test fold has ≥10 positives and ≥10 negatives (all 60 task-fold fits in Notebook 2 qualified).','Missing labels and eligibility.')
B(' "Pooled" = out-of-fold predictions pooled across folds per task, then macro-averaged over the 12 tasks. "Fold-mean" = macro-average within each fold, then mean ± SD across the 5 folds. The two must not be mixed.','Evaluation.')
B(' Notebook 2 compares chem, chem + true signatures, signatures only, and chem + row-permuted signatures (5 seeds: 101, 202, 303, 404, 505). Permutation keeps the signature distribution but breaks the compound link.','Permutation controls.')

H('Results: adult Tox21, 5-fold scaffold-grouped CV',10.5,5)
rows1=[
('Notebook 1 (pooled only)','','','',''),
('RF Morgan+desc','0.825','0.430','–','–'),
('XGB Morgan+desc','0.822','0.435','–','–'),
('XGB desc','0.822','0.427','–','–'),
('RF desc','0.818','0.417','–','–'),
('RF Morgan','0.780','0.343','–','–'),
('XGB Morgan','0.748','0.324','–','–'),
('Weighted MLP desc','0.772','0.351','–','–'),
('Weighted MLP Morgan+desc','0.736','0.309','–','–'),
('Notebook 2 (XGBoost; pooled and fold-mean)','','','',''),
('Chem only (Morgan+desc)','0.8224','0.4346','0.794 ± 0.051','0.380 ± 0.171'),
('Chem + true signatures','0.8254','0.4270','0.794 ± 0.052','0.366 ± 0.176'),
('Signatures only','0.8007','0.3878','0.766 ± 0.065','0.331 ± 0.179'),
('Chem + permuted (5 seeds, range)','0.774–0.781','0.332–0.338','0.738–0.745','0.285–0.293'),
('True signatures minus chem','+0.0030','−0.0076','−0.0003','−0.0135'),
]
table(['Arm','Pooled AUROC','Pooled AUPRC','Fold-mean AUROC','Fold-mean AUPRC'],rows1,[2.6,1.15,1.15,1.15,1.15],fs=8.5,center_from=1,keep=True)
P('Notebook 1 reports only pooled values for the full 12 tasks. Notebook 2 "chem only" is the same arm as Notebook 1 XGB Morgan+desc and reproduces it (0.822/0.435). Fold-mean scores are lower than pooled scores because folds vary a lot across scaffold groups (SD ≈ 0.05 AUROC), so I would quote both. Pediatric numbers are in Part II and are not comparable to this table. Source: values printed in the notebooks; no saved result files were available to cross-check (see provenance).',italic=True,size=8.5)

H('What the notebooks show',10.5,4)
B(' Standardized, scaffold-grouped evaluation gives a defensible adult baseline: about 0.82 pooled AUROC and 0.43 AUPRC. Descriptors carry most of the signal (descriptor-only is within 0.01 of the combined arm); Morgan alone is clearly weaker. The weighted MLP was the weakest family in CV.','What worked.')
B(' True signatures do not meaningfully improve on chemistry alone: ΔAUROC is +0.003 pooled and −0.0003 fold-mean, and ΔAUPRC is negative on both summaries. I ran no significance test on this adult comparison, so I make no significance claim. Real signatures beat the permuted controls (about +0.045 to +0.05 AUROC), but the permuted arms add 1,152 noise columns that degrade the model, so they are a floor, not a competitor. Signatures alone reach 0.80 AUROC, which suggests they encode information largely redundant with structure; I have not tested that directly.','What did not.')
H('Early pediatric FAERS probe inside the notebooks',10.5,4)
P('Notebook 1 (early block): 5,000 openFDA reports for ages 0 to 17 gave 137 compounds overlapping Tox21 after PubChem name-to-structure lookup. Models trained on a random 80/20 split with only 7 descriptors scored pediatric AUROC 0.53 (XGBoost), 0.44 (RF) and 0.54 (DNN), with MCC 0 (everything predicted in one class) and AUPRC ≈ 0.91 to 0.93, which reflects mostly-positive labels, not skill. Notebook 2 redid this with suspect-drug-only, latest-report-version filtering and standardized structures: 46 compounds, 42 positive and 4 negative (prevalence 0.913). Pediatric AUROC was 0.64 (chem) and 0.64 (chem + true signatures), but permuted arms ranged 0.55 to 0.67 and all paired-bootstrap CIs span zero (e.g., ΔAUROC +0.006, CI −0.38 to +0.49). With 4 negatives this is not a usable estimate. The later integrated run in Notebook 1 (39 drug names) returned 0 pediatric reports for every name, and its full-mode cell has no saved output. Notebook 2\'s later multisource pipeline (ClinicalTrials.gov + FAERS) crashed during signature generation, so it has no results. I report nothing from either.')

# ------------------------------------------------------------------ 4
H('4. Part II: subsequent experiments run on the box')
P('These experiments used pediatric endpoints, not Tox21 labels. Chemistry features are Morgan/ECFP4 and RDKit descriptors; ChEMBL target activity is the "biology" channel in E1 to E4 and E7. Source artifacts are listed in Section 8. Seeds 101 to 505 were used for permutation controls and bootstrap n = 2000 unless stated.',italic=True)

def EXP(title,verdict,parts):
    p=H(title,10.5,5)
    r=p.add_run('  |  Verdict: '+verdict); r.bold=True; r.font.size=Pt(9); r.font.color.rgb=RGBColor(0x8B,0x1A,0x1A)
    for lead,txt in parts: P(txt,lead,size=8.5)

EXP('E1. Aligned adult→pediatric transfer v2 (preliminary; CT.gov adverse-event incidence)','REJECTS multimodal benefit',[
('Question and data. ','Does adding a ChEMBL target-activity profile (chembl_target_profile; explicitly not Chemical Checker) to chemistry improve zero-shot adult→pediatric transfer of clinical-trial adverse-event incidence? Source: ClinicalTrials.gov interventional studies with posted results (75,437 fetched); 27,715 adult-only and 4,153 pediatric-only trials (mixed-age trials excluded); single-drug arms only. Name-to-structure mapping was stopped early once all 15 endpoints were eligible, leaving 434 adult and 927 pediatric structures (overlap 434) and 4,897 / 7,551 adult / pediatric structure-outcome labels. EudraVigilance was unavailable.'),
('Methods. ','Target = logit of smoothed incidence (a+0.5)/(n+1) per structure and harmonized endpoint (15 endpoints). Models trained on adult data only; pediatric labels never used for fitting. XGBoost (primary), RF, Ridge; hyperparameters tuned by adult scaffold-grouped CV and frozen. Arms: chem; bio-only; chem + true bio; chem + permuted bio (5 seeds). Pre-specified rule: true bio must beat chem and the permuted controls, with BH-corrected paired bootstrap.'),
('Results. ','Mean pediatric Spearman (XGBoost, 15 endpoints): chem 0.171, bio-only 0.120, chem + true bio 0.174, mean of permuted 0.172. True bio exceeded chem and permuted numerically in 9 of 15 endpoints, but no contrast survived BH correction (all p_BH ≥ 0.24). The only nominal p < 0.05 (ENDOCRINE_METABOLIC) had a negative Δ (−0.026, CI −0.048 to −0.004). Structure-seen pediatric rows scored higher than unseen (about 0.56 vs 0.40, pooled).')])

EXP('E2. Aligned transfer v3 (final; CT.gov adverse-event incidence)','REJECTS multimodal benefit (unchanged from E1)',[
('Question, data, methods. ','Same question and design as E1, with deferred adult-priority mappings completed and the early stop removed. Mapping: 17,014 names attempted, 9,664 resolved (56.8%), 1,216 unresolved, 6,134 excluded as non-drug. Final cohort: 2,856 adult and 835 pediatric structures (634 overlap); 23,602 adult and 6,734 pediatric structure-outcome labels; 15 endpoints. FAERS (25,000 + 25,000 openFDA reports) was used only as an external reporting-odds check, never merged into the incidence target.'),
('Results. ','Mean pediatric Spearman: chem 0.1244, bio-only 0.0850, chem + true bio 0.1296, mean permuted 0.1222. All arms fell versus E1 (chem 0.171 → 0.124). True − chem was positive in 10 of 15 endpoints but no BH-significant positive effect exists (smallest p_BH 0.015 belongs to a negative one): ANY_OTHER_AE Δ = −0.0905, 95% CI −0.1405 to −0.0398. The best positive nominal result, GASTROINTESTINAL Δ = +0.0387 (CI 0.0011 to 0.0768), has p_BH = 0.35. Pre-registered rule not met; the artifact states the null is valid and multimodal benefit must not be claimed.')])

EXP('E3. Independent read-only review of V3','primary REJECTS verdict preserved',[
('Method and results. ','Re-computed the V3 metrics and bootstrap from saved predictions: status MATCHED, 0 mismatches. Three critical findings. (C1) Structure-seen dominance: 634 of 835 pediatric structures appear in adult training; mean XGBoost Spearman is 0.178 on seen versus 0.020 on unseen structures (n = 201 unseen), so the headline ≈ 0.12 largely reflects same-drug continuity, not chemical generalization. On unseen structures all arms are near zero (chem 0.020, bio-only 0.037, chem + bio 0.039, permuted 0.023). (C2) The ChEMBL vocabulary has only 15 targets, mean per-target missingness is 92.4%, and 38.1% of compounds have any activity. (C3) The only BH-significant multimodal contrast is negative. Sensitivities: on the ChEMBL-covered subset chem 0.209 vs chem + true bio 0.223 (Δ +0.014); leave-one-endpoint-out true − chem stays between 0.003 and 0.012; true bio beats chem in 10 of 15 endpoints and all 5 permutations in 7 of 15. The review notes that the null is informative for large-n endpoints but underpowered for sparse organ classes, and that absence of benefit does not prove richer biology could never help.')])

EXP('E4. Ontogeny V4 (preliminary; age-conditioned enzyme features)','NO_SUPPORTED_BENEFIT; Phase 2 not run',[
('Question, data, methods. ','Static features cannot distinguish an adult from a pediatric version of the same drug. Does F = W(drug, enzyme) × O(enzyme, age), with published Hill/Rhodin-type maturation functions for 13 enzymes/transporters (e.g., CYP3A4, CYP2D6, UGT1A1, renal GFR), improve transfer over static chemistry? Data: V3 tables (2,856 / 835 structures; 15 endpoints). Substrate annotations came from curated FDA lists and ChEMBL CYP/UGT/CES activity proxies and are sparse (e.g., CYP3A4 annotated for 241 of 3,057 structures; CYP3A7 for 0). Four control families (permuted enzymes, permuted ages, adult-assigned ontogeny, permuted ontogeny rows; 5 seeds each). Five pre-specified criteria, all required.'),
('Results. ','Macro XGBoost Spearman: chem 0.1240; chem + ontogeny 0.1157 (Δ −0.0083); chem + exposure multiplier 0.1242; controls 0.116 to 0.119; ontogeny-only 0.022; static-metabolism-only 0.029. Only 5 of 15 endpoints had a positive Δ and none had a CI lower bound above 0; the BH-significant endpoint is negative (ANY_OTHER_AE Δ −0.086, CI −0.131 to −0.041). All five criteria false; by protocol no Phase 2 was run. The chem + ChEMBL-bio ontogeny variants had higher means (0.132 to 0.134), but that is the ChEMBL channel already tested in E2; the verdict file reports no BH-significant positive effect for the primary contrast. Descriptive age-band Spearman (chem vs chem + ontogeny): infant 0.298 vs 0.302 (n = 550), neonate 0.227 vs 0.232 (n = 198); no CI was read for these, so I make no claim from them.')])

EXP('E5. PACT: conformal intervals and abstention (CT.gov incidence)','SUPPORTED on its 6 interval-focused criteria; not a point-prediction claim',[
('Question, data, methods. ','Can a small pediatric calibration set turn an unreliable adult-trained model into one with valid empirical coverage and useful intervals? Protocol frozen 2026-09-25 before fitting uncertainty models. V3 data, 15 endpoints; structure-grouped 20/20/60 split: 167 method-development, 167 calibration, 501 final-test structures (4,035 test rows). Adult XGBoost point model (macro Spearman 0.1244, reproducing V3). Compared: global split conformal (B4), endpoint-Mondrian (B5), denominator-scaled (B6), and PACT (localized, scale-normalized; empirical, no formal guarantee under shift). Nominal coverage 90%.'),
('Results. ','Final-test coverage: PACT 0.902, global 0.923, Mondrian 0.928, denominator-scaled 0.906. Mean logit interval width: PACT 5.251 vs global 6.083 (Δ −0.832, 95% CI −0.935 to −0.726; structure bootstrap n = 5000) and vs Mondrian 6.322. Ten repeated splits: PACT coverage 0.897 ± 0.009, width 4.960 ± 0.129. Selective MAE at 50% retained: 1.188 (primary abstention) vs 1.293 random, but E6 later showed this equals the ensemble-disagreement ranking and equals the global method (Δ = 0), so it is not a PACT-specific gain. Negative controls: shuffled method-dev labels gave coverage 0.927, width 6.290; shuffled Tanimoto gave coverage 0.918, width 5.388. Leakage checks passed. The artifact states that the point model stays weak (Spearman ≈ 0.12) and that no improvement in point prediction or formal coverage guarantee is claimed.')])

EXP('E6. PACT validation v2','reproduction passed; efficiency advantage inconclusive; external validity not tested',[
('Method and results. ','Audit of E5 on the same (already inspected) final-test set, so all new claims are exploratory. Reproduction passed (e.g., coverage 0.902354 vs 0.902; width 5.251112 vs 5.251). Added baselines: B7 individualized residual-scale conformal (same scale features, no locality), B8 scale + Mondrian, B9 ensemble-only. B7 was selected as the strongest fair comparator using calibration data only. PACT vs B7: width 5.251 vs 5.388 (Δ −0.136, CI −0.185 to −0.085); coverage 0.902 vs 0.915 (Δ −0.013, CI −0.017 to −0.008). The prespecified "meaningful" threshold (CI entirely below −0.10) was not met. Repeated splits: PACT 4.974 ± 0.138 vs B7 4.998 ± 0.125 (practically tied); descriptive matched-coverage widths are identical for PACT, B7 and B8 (5.107). PACT coverage at nominal 80/90/95%: 0.805/0.902/0.956; sparse subgroups stay near nominal (low-denominator 0.906, structure-unseen 0.926, rare endpoint 0.905), with wider intervals on unseen structures. Width-based abstention at 50%: PACT 0.944 vs B7 0.927, so no PACT-specific selective claim. No compatible independent pediatric CT.gov cohort was found. Artifact summary: suitable as a careful empirical study, not a new-guarantee method paper.')])

EXP('E7. REACT-Fuse (reliability-gated biology fusion; CT.gov incidence)','NOT_SUPPORTED (pre-registered STOP/GO failed)',[
('Question, data, methods. ','Does endpoint-matched, reliability-gated biology (ChEMBL measured activity, PubChem assay counts, Open Targets, pathway themes, V4 ontogeny; ToxCast/Tox21 and LINCS were not fetched) improve few-shot transfer beyond chemistry? Same V3 cohort (2,856 / 835; overlap 634). Pre-registered gate: proceed to full fusion only if ≥ 40% of pediatric structures have useful biology and real (chemistry-orthogonalized) biology predicts adult out-of-fold chemistry residuals better than shuffled, missingness-only and confidence-only controls.'),
('Results. ','Coverage passed (474 of 835 pediatric structures, 56.8%). Residual gate failed: macro Spearman real −0.0477, shuffled −0.0454, missingness-alone −0.0491, confidence-alone −0.0491; real − shuffled Δ −0.0023 (CI −0.0082 to +0.0035). Biology columns were not predictable from chemistry (median out-of-fold R² −1.14; 0 of 31 columns above 0.1). Full point-prediction fusion was therefore not run; criteria c1 to c6 all false. Chem XGBoost baseline on this split: 0.1367 (reference 0.124). Lightweight A to I diagnostics run under STOP are labelled exploratory and did not meet the strict criteria. Secondary uncertainty check on its own split: global conformal coverage 0.886, width 5.495 vs "PACT-bio" coverage 0.891, width 10.314, so biology as an uncertainty scale gave much wider intervals.')])

EXP('E8. PACT-AL: conformal active learning (CT.gov incidence)','PARTIALLY_SUPPORTED',[
('Question, data, methods. ','At the same labeling budget, does acquisition by PACT uncertainty + chemical diversity + endpoint under-representation (λ = 1/3 each) beat random and other strategies? A temporal hold-out of new post-V3 trials was unavailable, so design = 5-fold structure-grouped CV (results summary: 2,610 adult / 808 pediatric structures, overlap 597), 10 seeds = 50 runs; 5% initial labels, 10% fixed calibration, budgets 1% to 40% of the pool; nine strategies A to I (random, uncertainty, ensemble disagreement, chemical novelty, diversity, endpoint-need, U+D, full PACT-AL, oracle). Metric: AULC of endpoint-macro Spearman. Retrospective simulation only.'),
('Results. ','AULC: random 0.0956, PACT-AL 0.1011, endpoint-need alone 0.1155 (best), diversity 0.0763, oracle 0.0860. PACT-AL − random = +0.0056 (CI −0.0025 to +0.0133, p = 0.17): directionally positive, not distinguishable from zero. PACT-AL − endpoint-need = −0.0144 (CI −0.0195 to −0.0091), so the simplest strategy beat the full method; PACT-AL − diversity +0.0248 and − U+D +0.0059 (CI 0.0019 to 0.0100). Endpoint-need reached random\'s 40%-budget peak at 20%; the PACT-AL mean curve did not exceed it within the budgets. Empirical coverage stayed near 0.93 (nominal 0.90). No formal guarantee under adaptive selection; PACT final-test not reused as confirmation.')])

EXP('E9. DART: continuous-age FAERS trajectories','PARTIALLY_SUPPORTED (4 of 9 frozen criteria), but below the best baseline',[
('Question, data, methods. ','Can a model with continuous age and hierarchical shrinkage predict age-specific drug-event reporting signals better than age-group baselines? Data: FAERS 2004 to 2026Q2 downloaded and parsed (6,987,407 reports with valid age); modeling restricted to 2018+, primary-suspect single-ingredient drugs, top 400 drugs × 400 events, medium 300 × 300 scale (train 159,130, dev 62,987, test 103,940 cells). Frozen temporal split: train through 2022, dev 2023, test 2024 onward. Target: empirical-Bayes-shrunk log reporting odds ratio, which is a reporting association, not incidence. Baselines B1 to B10; primary metric reliability-weighted macro Spearman; criteria frozen before final-test inspection.'),
('Results. ','Best baseline B4 (separate age-group) 0.664; DART 0.531; DART − B4 = −0.0839 (CI −0.0982 to −0.0703). Others: B8 0.650, B5 0.650, B9 0.650, B2 0.637, B1 0.637, B7 0.596, B10 0.518 (an XGBoost approximation of literature ML), B3 0.222, B6 0.170. Criteria passed: seed consistency, shuffled-age control, calibration (0.031 vs 0.275 as printed), not driven by top drugs (Spearman 0.590 vs 0.588 without them). Criteria failed: beats baseline, paired CI, neonatal/infant benefit, unseen/temporal gain (unseen-drug slice was empty by construction), known-signal recovery. Per-stage Spearman (global, metrics_by_age.csv): infants DART 0.524 vs B4 0.424 (n = 361) but neonates 0.094 vs 0.274 (n = 112). Note: criterion 5 is marked pass although the shuffled-age advantage is NaN in the verdict file, so I would treat that criterion as unverified.')])

EXP('E10. CCAD: conformal alarms and selective forecasting (FAERS signals)','NOT_SUPPORTED (pilot stop; final test not opened)',[
('Method and results. ','Question: can age-Mondrian conformal residual alarms (Module A, monitoring) and width-based pre-outcome abstention (Module B, selective) work on quarterly FAERS signals? Data: 34 quarters (2018Q1 to 2026Q2), 980,709 quarter-level signal rows in train/point-val/calibration/monitor-dev/final-test blocks; B4 as point predictor (independent reproduction wSpearman 0.6625 vs 0.6640). Frozen pilot rule: point OK and (monitoring OK or selective OK). Point check passed (0.778 ≥ 0.40). Monitoring failed: monitor-dev had only 2 quarters, ~24 labeled transitions and 1 event, so precision at 20% false-alarm rate was 0.0 for both CCAD and the rolling baseline. Selective was a near miss: MAE at 75% retention 0.5030 vs random 0.5213 (Δ 0.0183 < the frozen 0.02 margin). The final test was not opened and criteria were not weakened. The artifact frames this as an honest negative / feasibility audit.')])

EXP('E11. SCSP: stage-constrained selective prediction (FAERS signals)','NOT_SUPPORTED (exploratory); confirmatory NOT_RUN',[
('Method and results. ','Question: on top of the B4 predictor, does an instance-scaled, age-calibrated abstention score (abstain on predicted interval width; equal retention per age stage) beat the strongest fair baseline in MAE at 75% retention? Data: DART panels; train 157,252, dev 62,956, test 103,940 cells; 297 drugs × 298 events; B4 verification wSpearman 0.6625. The DEV gate failed: the proposed score correlated ρ = 0.997 with a global individualized scale (adding negligible ranking signal) and did not beat it (ΔMAE 0.0000 vs a −0.02 threshold). Exploratory test (2024 to 2026Q2, a reused period): MAE 0.5401 for both, Δ 0.0000, CI [0, 0], p = 1.0; alternative regime Δ −0.00007 (CI −0.0004 to +0.0005). Empirical coverage of the proposed intervals was 0.864 vs 0.890 for age-Mondrian (nominal 0.90). Neonate and infant test cells: 112 and 361. No unused later FAERS quarter exists, so the frozen confirmatory protocol was not run. The SCSP notebooks in the folder are 2- to 3-cell stubs that print the verdict, not executed analyses.')])

EXP('E12. Adaptive pediatric borrowing (FAERS signals)','NOT_SUPPORTED; stopped by the headroom gate',[
('Method and results. ','Question: can a rule using only prediction-time information tell when borrowing adult reporting history improves on B4 for sparse pediatric cells (prior pediatric count ≤ 5)? Protocol and gate thresholds were frozen before measurement. Dev sparse subset: n = 1,666 (1,538 drug-event pairs). MAE: B4 1.3791, adult-informed 1.3652 (gain 0.0139; required ≥ 0.02, failed); oracle 1.3075 (gain 0.0717; headroom check passed). Both conditions are required, so the gate decision was STOP_NO_DEMONSTRATED_OPPORTUNITY and no adaptive method was trained. For documentation only, exploratory test results (reused 2024 to 2026Q2) show a larger gap on the sparse subset (B4 1.173 vs adult-informed 1.131, n = 5,719; Spearman 0.538 vs 0.537) and on all pediatric cells (MAE 0.924 vs 0.890). The artifact explicitly does not use these to reopen the gate, because that would be outcome-dependent; I treat them only as a hypothesis for independent data.')])

EXP('E13. SAFE-Ped (frozen B4 + age-fair conformal selective abstention)','cannot be verified from the box',[
('Status. ','My notes say its dev gate, evaluation, bootstrap, notebooks, ZIP and verdict were completed. I searched /workspace, /workspace/tox21 and the user directories for a SAFE-Ped folder, files, or text mentions (including inside deliverable ZIP listings) and found none. The closest existing artifacts are SCSP (E11) and CCAD (E10), which share the frozen-B4 design. Because no verdict file exists on the box, I report no SAFE-Ped numbers and no verdict.')])

EXP('A1. ToxLens baseline reproduction (adjacent; external benchmark)','reproduction; extension not run',[
('Method and results. ','Independent re-run of four ECFP4 baselines (RF, XGB, MLP, SVM) from the published ToxLens paper on its frozen splits (train 21,578; validation 2,798; test 2,971) over 11 endpoints, validation-MCC thresholds. Macro test AUROC (ours vs published): RF 0.7806 vs 0.7796, XGB 0.7426 vs 0.7432, MLP 0.6939 vs 0.6938, SVM 0.6627 vs 0.6627; macro test MCC averaged over models 0.2639 vs 0.2619. The extension (hold-out calibration, K splits) was a dry-run skeleton only, with no extension metrics. This benchmark differs from my Tox21 CV (different splits and endpoints), so I do not compare its numbers to Part I.')])

# ------------------------------------------------------------------ 5
H('5. Cross-experiment interpretation (candid)')
B(' Interval validity and efficiency: empirical coverage near 90% with intervals narrower than global conformal (E5; reproduced in E6). Honest evaluation machinery: pre-registered gates stopped four lines early (E4, E7, E10, E12; the E11 dev gate also failed) instead of continuing to search, and an independent review reproduced the V3 numbers exactly (E3).','What worked.')
B(' Multimodal biology never beat chemistry plus permuted controls on pediatric transfer (E1, E2, E3, E7; the Tox21-signature comparison in NB2 is also a tie). Age-dependent enzyme features did not help (E4). The continuous-age FAERS model was below the strongest simple baseline (E9). Selective abstention and adaptive borrowing showed no demonstrated gain over fair baselines (E10, E11, E12). PACT\'s advantage over the strongest fair interval baseline is small and not protocol-meaningful (E6). Active learning was directionally positive but not significant, and endpoint-availability alone beat the full method (E8).','What did not.')
B(' All pediatric experiments after the notebooks use CT.gov adverse-event incidence or FAERS reporting signals with chemistry and ChEMBL features. Based on my search, Tox21 assay labels are not used in any of them (the string appears only in folder paths, loose project labels in some reports, and as a documented unfetched source in E7). Tox21 assays are in-vitro pathway readouts, and the notebooks show only that a model predicts them for scaffold-held-out compounds. The one place where Tox21-derived models were linked to pediatric outcomes was the notebook probe, which had 46 to 137 compounds, 4 negatives, and uninformative results. So adult Tox21 performance establishes neither adult clinical performance nor adult→pediatric transfer.','Why adult Tox21 does not establish adult→pediatric transfer.')
B(' The zero-shot pediatric Spearman is about 0.12 and is mostly same-drug continuity: structure-seen 0.178 vs structure-unseen 0.020 (E3). The ChEMBL channel is sparse (15 targets, ~92% missing), annotation coverage for enzymes is under 8% for CYP3A4 and lower for others (E4), and the FAERS B4 baseline is already strong at 0.664 (E9). In E12 the adult-borrowing gain was 0.0139 against a 0.02 requirement, while oracle headroom was 0.072. My reading, which I have not tested, is that a strong simple baseline leaves little room for added structure. In E10, the available monitoring window was too short to evaluate alarms at all.','Headroom and ceiling issues.')
B(' The artifacts support one forward step: any confirmatory claim needs independent data, i.e., later unused FAERS quarters or an independent pediatric cohort (SCSP\'s frozen confirmatory protocol requires this; PACT validation found no compatible cohort). CCAD lists a longer monitoring window and cell-level streams as options for a retry. I have no SAFE-Ped artifact to cite, so I propose nothing from it.','Next step supported by the artifacts.')

# ------------------------------------------------------------------ 6
H('6. Limitations (as found in the notebooks and artifacts)')
B('Notebook FAERS probe: the label is the fraction of a drug\'s reports flagged serious (≥ 0.5), not an incidence; there is no exposed-child denominator and no adjustment for age, indication or co-medication; the flag is report-level (Notebook 1 gave it to concomitant drugs too). Cohorts are tiny and imbalanced (137 or 46 compounds; 4 negatives).')
B('Endpoint mismatch throughout: Tox21 in-vitro activity vs CT.gov incidence vs FAERS reporting associations (EB-shrunk log ROR, which is not incidence or causal risk). The notebook "transfer gap" compares different tasks and metrics and is not like-for-like.')
B('CT.gov data (E1 to E8): incidence is a smoothed reporting construct with max-within-arm aggregation and a 60-term keyword fallback for MedDRA classes; confounding by indication remains; only 56.8% of names mapped; the ChEMBL vocabulary is 15 targets; EudraVigilance was unavailable; enzyme annotations are sparse (E4).')
B('FAERS data (E9 to E12): spontaneous reporting bias; primary-suspect 2018+ only; top 400 × 400 vocabulary; period-aggregated panels; B10 is an approximation of literature ML; neonate/infant cells are small (112 and 361 in the test period).')
B('No independent pediatric dataset: PACT validation lists external validity as not tested; no post-V3 CT.gov trials and no unused later FAERS quarter exist on the box. Final-test sets were reused exploratorily: PACT\'s test set in E6 (and E7\'s secondary check was run on its own split), and DART\'s 2024 to 2026 block in E10 to E12. New claims on reused data are exploratory by the artifacts\' own labels.')
B('Conformal guarantees are not claimed under adult→pediatric or temporal shift, or under adaptive selection; coverage is empirical. Many representations, controls and endpoints per experiment mean exploratory-risk if misread (E4 limitation). Cohort counts differ slightly between artifacts (E8 reports 2,610 / 808 structures vs 2,856 / 835 in V3).')
B('Notebook numbers come from printed outputs only; I did not read the Drive result files, and I ran no significance test on the adult Tox21 comparison.')

# ------------------------------------------------------------------ 7
H('7. Questions for you')
P('1. Scientific direction. Multimodal biology has been null in every pediatric-transfer test (E1 to E4, E7), while the uncertainty work (E5 to E6) is reproducible but only modestly better than a strong baseline. Should I (a) reposition the project as a careful negative-result and audit study, (b) pursue the uncertainty/abstention line, but only on independent data, or (c) change the question?')
P('2. Pediatric outcomes and data. Which pediatric outcomes are appropriate, and is there an independent dataset (with exposure denominators or curated outcomes, or later unused FAERS quarters) that we can use for confirmation instead of reused test sets and FAERS reporting fractions?')
P('3. MOF aim. Neither notebook nor any box experiment contains MOF work. I searched /workspace/tox21, the aligned v2/v3 folders, the V3 review, and both notebook dumps for "MOF" and "metal-organic". The only hits are inside raw downloaded ClinicalTrials.gov JSON pages (e.g., a MeSH term "ZIF-8 metal-organic framework" in a trial record and "MOF" meaning major osteoporotic fracture in another), which are data records, not project work. Do we keep the MOF aim (which would need new data and a separate design) or drop it from this project?')

# ------------------------------------------------------------------ 8
H('8. Provenance and verification notes')
P('Notebook numbers come from outputs printed in two Colab notebooks (Notebook 1: adult baselines, scaffold CV and early FAERS probe, 42 cells; Notebook 2: signature and permutation comparison plus the pediatric probe, 10 cells), not from Drive CSVs. Notebook 2 saved results to Google Drive (Tox21_multimodal_checkpoints/27f4dc5e18e0/ and pediatric_transfer/cohort_59e0f246ccc3/), which I could not read, so I did not cross-check against them. [Fill in actual notebook filenames here.] ')
P('Not used: Notebook 1\'s "quick mode" pooled means (about 0.81 to 0.82 AUROC) appear to average only the two displayed tasks and rely on helper scripts that are not in the notebook, so they are not comparable to the 12-task values. Notebook 1 also contains a hard-coded openFDA API key in its source (cell 30); it should be removed and rotated before the notebook is shared. Copies of an openFDA key file also sit in a few box project folders; they should be excluded from any archive that is shared. The key is not reproduced in this document.')
P('Part II numbers come from these read-only artifacts (verdict files first, then results files and reports; nothing was re-run):',size=9.5)
prov=[
('E1','/workspace/pediatric_transfer_aligned_v2/results/hypothesis_verdict.json; reports/results_summary.md; reports/methods.md; results/bootstrap_comparisons.csv'),
('E2','/workspace/pediatric_transfer_aligned_v3/results/hypothesis_verdict.json; results/conclusion_delta.json; reports/results_summary.md; reports/comparison_vs_preliminary.md'),
('E3','/workspace/pediatric_transfer_v3_independent_review/results/audit_findings.json; reproduction_delta.json; sensitivity_summary.json; uncertainty_summary.json; reports/01_audit_report.md'),
('E4','/workspace/tox21/pediatric_transfer_ontogeny_v4_preliminary/hypothesis_verdict.json; results_summary.md; methods.md; paired_bootstrap_comparisons.csv; age_band_results.csv; data_coverage.csv'),
('E5','/workspace/tox21/pact_pediatric_conformal_v1/results/hypothesis_verdict.json; README.md; reports/results_summary.md; config/frozen_protocol.md; results/negative_controls.csv'),
('E6','/workspace/tox21/pact_validation_v2/results/hypothesis_verdict.json; reports/results_summary.md; reports/limitations.md; results/repeated_split_summary.csv'),
('E7','/workspace/tox21/react_fuse_data_aligned_v1/results/hypothesis_verdict.json; reports/results_summary.md; reports/methods.md; results/coverage_audit.json; residual_predictability_audit.json; biology_from_chem_redundancy.json'),
('E8','/workspace/tox21/pact_active_learning_v1/results/hypothesis_verdict.json; reports/results_summary.md; results/aulc_summary.csv; bootstrap_comparisons.csv'),
('E9','/workspace/tox21/dart_developmental_pharmacovigilance_v1/results/hypothesis_verdict.json; primary_metrics.csv; metrics_by_age.csv; scale.json; cohort_flow.json; reports/results_summary.md'),
('E10','/workspace/tox21/ccad_real_faers_v1/results/hypothesis_verdict.json; pilot_stop_go.json; sample_sizes.json; reports/feasibility_null_report.md'),
('E11','/workspace/tox21/stage_constrained_selective_prediction_v1/results/hypothesis_verdict.json; dev_gate.json; score_duplication_check.json; reports/results_summary.md'),
('E12','/workspace/tox21/adaptive_pediatric_borrowing_v1/results/hypothesis_verdict.json; headroom_gate.json; reports/results_summary.md'),
('E13','No artifact found (searched /workspace, /workspace/tox21 and user directories); no verdict file located.'),
('A1','/workspace/toxlens_repro/docs/REPRODUCTION_REPORT.md; outputs/comparison_summary.json'),
]
table(['ID','Artifact paths'],prov,[0.45,6.85],fs=7,section_rows=False)
P('Other folders in /workspace (e.g., medai_repro, mfcr-experiment, cac_algorithm_evaluation, ieee-nam-*, pranav-datasets, pranav-opportunities, pranav-lab-outreach) belong to unrelated projects and are not included.',italic=True,size=8.5)
d.save('Tox21_PI_update_full.docx')
