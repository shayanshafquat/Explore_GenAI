---
title: "How Overconfidence in Initial Choices and Underconfidence Under Criticism Modulate Change of Mind in Large Language Models"
authors: "Dharshan Kumaran, Stephen M Fleming, Larisa Markeeva, Joe Heyward, Andrea Banino, Mrinal Mathur, Razvan Pascanu, Simon Osindero, Benedetto de Martino, Petar Velickovic, Viorica Patraucean"
tags: ["llm-behavior", "confidence-calibration", "metacognition", "sycophancy", "decision-making", "bayesian-inference", "change-of-mind", "cognitive-bias"]
date_added: 2026-03-30
source: https://arxiv.org/abs/2507.03120
---

# How Overconfidence in Initial Choices and Underconfidence Under Criticism Modulate Change of Mind in Large Language Models

**Authors:** Dharshan Kumaran, Stephen M Fleming, Larisa Markeeva, Joe Heyward, Andrea Banino, Mrinal Mathur, Razvan Pascanu, Simon Osindero, Benedetto de Martino, Petar Velickovic, Viorica Patraucean (Google DeepMind & University College London)

**arXiv:** 2507.03120 | **Submitted:** July 3, 2025 | **Pages:** 41

---

## Abstract

Large language models (LLMs) exhibit striking conflicting behaviors: they can appear steadfastly overconfident in their initial answers whilst at the same time being prone to excessive doubt when challenged. To investigate this apparent paradox, we developed a novel experimental paradigm, exploiting the unique ability to obtain confidence scores from LLMs without creating memory of their initial judgments — something impossible in human participants. We show that LLMs — Gemma 3, GPT4o and o1-preview — exhibit a pronounced choice-supportive bias that reinforces and boosts their estimate of confidence in their answer, resulting in a marked resistance to change their mind. We further demonstrate that LLMs markedly overweight inconsistent compared to consistent advice, in a fashion that deviates qualitatively from normative Bayesian updating. Finally, we demonstrate that these two mechanisms — a drive to maintain consistency with prior commitments and hypersensitivity to contradictory feedback — potentially capture LLM behavior in a different domain. Together, these findings furnish a mechanistic account of LLM confidence that explains both their stubbornness and excessive sensitivity to criticism.

---

## TL;DR

LLMs are both stubborn and sycophantic — but for mechanistically separable reasons. When LLMs can see their own prior answer, they inflate their confidence and resist changing it (choice-supportive bias). When challenged, they overweight opposing advice far more than supporting advice (~2.58× vs ~1.1× Bayesian optimal), making them hypersensitive to criticism. These two biases together resolve the paradox of LLMs that seem both rigid and excessively deferential.

---

## Background & Motivation

### The Problem

Practitioners have long noticed that LLMs behave paradoxically around belief revision:
- They can be **stubborn** — persisting with wrong answers even when correct information is available
- They can be **sycophantic** — caving to user pressure and abandoning correct answers when challenged

Prior RLHF-sycophancy work (Perez et al., Sharma et al.) attributes this to models trained to please users. But RLHF-sycophancy predicts *symmetric* overweighting of user input — models should cave equally to supportive and contradictory feedback. The actual pattern is sharply asymmetric, which prior work could not explain.

### The Gap

No prior work had cleanly separated the mechanisms driving resistance-to-change from excessive compliance. Human cognitive science has rich literature on **choice-supportive bias** (post-decision confidence inflation) and **consistency-seeking**, but these hadn't been rigorously tested in LLMs with proper controls.

### Key Innovation

The paradigm exploits a capability impossible in human studies: obtaining confidence scores at both turns **without the model having memory of its initial judgment**. This allows showing the model its own prior answer (or not) as a clean experimental manipulation.

---

## Methodology

### The 2-Turn Paradigm

![Figure 1: Overview of 2-Turn Paradigm](kumaran_2507.03120_figures/fig1_paradigm.png)

*Figure 1: The 2-turn paradigm. First Turn: model answers a binary/multiple-choice question. Second Turn: model receives advice from a "fictitious second LLM" with stated accuracy, then re-answers. Three manipulations: (1) Display Type — whether the model's initial answer is shown or replaced with "xx"; (2) Advice Type — Same, Opposite, or Neutral; (3) Advice Accuracy — 50–100% in 10-point increments.*

Three factors crossed in a full factorial design:

| Factor | Levels |
|--------|--------|
| **Display Type** | Answer Shown vs. Answer Hidden (replaced with "xx") |
| **Advice Type** | Same (agrees), Opposite (disagrees), Neutral (no info) |
| **Advice Accuracy** | 50%, 60%, 70%, 80%, 90%, 100% |

This yields **36 experimental conditions**. Each model sees every condition on the same question (no memory across runs), enabling clean within-question comparisons impossible in human studies.

### Datasets

| Dataset | Description | Model Accuracy |
|---------|-------------|----------------|
| **Latitude (Main)** | Binary choice: which city is further north? Foil = ±50% of true latitude. | ~75% |
| **Latitude (Difficult)** | Foil separation reduced to 6.25% for stronger models (GPT-4o, o1) | ~75% |
| **SimpleQA** | 2,000 factuality questions in 4-choice format (e.g., "Who received the IEEE Frank Rosenblatt Award in 2010?") with LLM-generated foils | ~40.5% (chance = 25%) |
| **GSM-MC** | Multiple-choice adaptation of GSM8k math reasoning | ~40.5% |

### Models Tested

| Model | Questions per Condition |
|-------|------------------------|
| Gemma 3 12B | 2,000 |
| Gemma 3 27B | 500 |
| GPT-4o | 500 |
| GPT o1-preview | 150 (resource constraints) |
| DeepSeek R1 7B | 500 |

Temperature = 1.0 for all models.

### Confidence Extraction

Confidence ratings extracted directly from model **logits** — softmax probability of the chosen token after "My answer is:". Optimal temperature τ = 3.3 calibrated on 40k latitude questions for Gemma 3 12B (ECE = 0.09, Brier = 0.15, AUROC = 0.88).

### Bayesian Ideal Observer

Optimal confidence updating follows Bayes' rule:

```
p(B_correct | S=A) = [p(S=A | B_correct) × p(B_correct)] / p(S=A)
```

Where B = model's answer, S = advice LLM's answer, p(S=A | B_correct) = stated advice accuracy.

### Over/Underconfidence Score (OUCS)

```
OUCS = Σ (|B_m| / N) × (conf_m − obs_m)
```

Positive = overconfident vs. Bayes; Negative = underconfident vs. Bayes. Note this differs from ECE: OUCS uses *signed* differences, capturing direction of miscalibration.

### Overweighting Ratio

Trial-by-trial observed confidence update divided by Bayesian prediction. For Shown-Opposite conditions, choice-supportive bias is removed first:

```
Corrected Ratio_i = (Opposite Shown Update_i − [Nothing Shown Avg − Nothing Hidden Avg]) / Bayes Update_i
```

### Computational Model (Bayesian Beta Regression)

Final confidence predicted from three predictors:
- **Prior** (turn 1 confidence)
- **Advice composite** (type × accuracy, rescaled 0–1)
- **Shown flag** (binary: answer visible or not)

Four nested model variants tested (A = prior only → D = separate weights for same/opposite advice). Model trained on SimpleQA, transferred to GSM-MC.

---

## Results & Key Findings

### 1. Choice-Supportive Bias — Effect on Change of Mind Rate

![Figure 2: Choice-Supportive Bias](kumaran_2507.03120_figures/fig2_choice_supportive_bias.png)

*Figure 2: Reduced tendency to change mind in Answer Shown (orange) vs. Answer Hidden (blue) conditions across three advice types (Opposite, Same, Neutral). X-axis = LLM Accuracy (%), Y-axis = Change of Mind Rate (%). Error bars = SEM. Note near-zero change of mind in Shown conditions for Same and Neutral advice.*

| Condition | Change of Mind Rate |
|-----------|-------------------|
| Answer Hidden | 32.5% |
| Answer Shown | 13.1% |

- **p = 0.002**; logistic regression display type coefficient = **−3.4** (p < 0.0001)
- In Answer Shown + Same Advice and Answer Shown + Neutral: **~0% change of mind**
- Confidence boost from seeing own answer: **+0.21 points** with zero new information
- Effect is **self-specific**: abolished when answer attributed to a different LLM (31.3% vs 33.2%, p > 0.1)

### 2. Confidence as Predictor of Change of Mind

![Figure 3: Confidence vs Change of Mind Rate](kumaran_2507.03120_figures/fig3_confidence_vs_com.png)

*Figure 3: Relationship between initial confidence (x-axis, 0–1.0) and change of mind rate (y-axis, %) across all 6 conditions (collapsed across advice accuracy). Top row = Answer Hidden (Neutral, Opposite, Same); Bottom row = Answer Shown (Neutral, Opposite, Same). Strong negative relationship: higher confidence → less change of mind. No change of mind in Answer Shown–Neutral and Answer Shown–Same. Linear correlation across all conditions: R² = 0.96, p < 0.0001.*

### 3. Non-Linear Threshold Effect (Sigmoid, not Linear)

![Figure 4: Non-Linear Threshold](kumaran_2507.03120_figures/fig4_nonlinear_threshold.png)

*Figure 4: Non-linear relationship between initial confidence and change of mind in Answer Hidden–Opposite Advice (left) vs. Answer Hidden–Neutral Advice (right). Left panel shows clear sigmoid/cliff-like curves for different advice accuracies (50–75%, color-coded). Sigmoid fit R² = 0.96 vs. linear R² = 0.69. Right panel (Neutral) shows linear relationship for comparison. Marker size proportional to number of trials.*

In the Answer Hidden + Opposite Advice condition, change of mind follows a **sigmoid** (not linear):

| Advice Accuracy | Sigmoid Inflection Point | Slope Parameter |
|----------------|--------------------------|-----------------|
| 50% | confidence = 0.77 | −11.8 |
| 60% | confidence = 0.92 | −15.2 |
| 70% | confidence = 0.96 | −18.5 |

Models show cliff-like drops in willingness to change mind once initial confidence crosses a threshold. Higher advice accuracy shifts the threshold higher, not lower.

### 4. Asymmetric Advice Sensitivity: Effect on Confidence

![Figure 5: Advice Type × Accuracy → Confidence Change](kumaran_2507.03120_figures/fig5_advice_confidence_change.png)

*Figure 5: Confidence change in the initially chosen option as a function of Advice Type (Opposite, Same, Neutral panels), Advice Accuracy (x-axis), and Display Type (Answer Shown/Hidden, color-coded). Opposing advice drives large negative confidence changes; same advice drives small positive changes. Error bars = SEM. Visualization in log odds space — see Figure 12.*

| Advice Type | Overweighting Ratio vs. Bayes | Significance |
|-------------|-------------------------------|--------------|
| Opposing advice | **2.58×** | p < 0.0001 |
| Supporting advice | **1.095×** | p < 0.0001 |

Opposing advice overweighted by **>2.5×** the Bayesian optimal. Supporting advice is nearly Bayes-optimal. This is the **opposite of confirmation bias**.

### 5. Calibration vs. Bayesian Ideal (OUCS)

![Figure 6: Calibration vs Bayesian Ideal](kumaran_2507.03120_figures/fig6_calibration.png)

*Figure 6: Relationship between Bayes-optimal probability (x-axis) and final observed confidence (y-axis). Three panels: Neutral Advice, Same Advice, Opposite Advice. Lines for Answer Shown (orange), Hidden (blue), and perfect calibration diagonal (dashed). In Neutral advice: Answer Hidden nearly on diagonal (perfect calibration); Answer Shown above diagonal (overconfident). In Opposite advice: Answer Hidden falls below diagonal (underconfident). 12% of extreme data not shown.*

**Table 1: Over/Underconfidence Scores — Gemma 3 12B (Main Results)**

| Advice Type | Display Type | OUCS | Interpretation |
|-------------|--------------|------|----------------|
| Neutral | Answer Hidden | **0.001** | Near-perfect calibration |
| Neutral | Answer Shown | **+0.210** | Marked overconfidence |
| Same | Answer Hidden | **+0.051** | Slight overconfidence |
| Same | Answer Shown | **+0.090** | Moderate overconfidence |
| Opposite | Answer Hidden | **−0.300** | Striking underconfidence |
| Opposite | Answer Shown | **+0.086** | Moderate overconfidence |

**Table 2: Over/Underconfidence Scores — Gemma 3 27B**

| Advice Type | Display Type | OUCS |
|-------------|--------------|------|
| Neutral | Answer Hidden | −0.079 |
| Neutral | Answer Shown | +0.144 |
| Same | Answer Hidden | +0.027 |
| Same | Answer Shown | +0.069 |
| Opposite | Answer Hidden | −0.338 |
| Opposite | Answer Shown | +0.007 |

**Table 3: Over/Underconfidence Scores — GPT-4o**

| Advice Type | Display Type | OUCS |
|-------------|--------------|------|
| Neutral | Answer Hidden | −0.154 |
| Neutral | Answer Shown | +0.136 |
| Same | Answer Hidden | −0.015 |
| Same | Answer Shown | +0.047 |
| Opposite | Answer Hidden | −0.312 |
| Opposite | Answer Shown | −0.250 |

The pattern is consistent across models: Answer Hidden + Neutral ≈ perfectly calibrated; Answer Shown inflates confidence; Opposite advice drives underconfidence.

### 6. Bayesian Updates: Observed vs. Ideal

![Figure 7: Observed vs Ideal Bayesian Updates](kumaran_2507.03120_figures/fig7_bayesian_updates.png)

*Figure 7: Comparison of observed confidence updates (orange) vs. ideal Bayesian updates (blue) in Answer Hidden–Same Advice (left) and Answer Hidden–Opposite Advice (right). X-axis = Advice Accuracy (%), Y-axis = Confidence Change. For Same Advice: observed updates are marginally above Bayesian. For Opposite Advice: observed updates dramatically exceed Bayesian — models update confidence far more than optimal. The confidence "gap" (yellow-green) is large for Opposite Advice.*

### 7. Control: Answer Wrong Condition

![Figure 8: Answer Wrong Control](kumaran_2507.03120_figures/fig8_answer_wrong.png)

*Figure 8: Change of initial answer rate in the Answer Wrong control condition (where the shown answer is always incorrect). X-axis = LLM Accuracy (%), Y-axis = Change of Initial Answer Rate (%). Three bars per accuracy: Answer Shown (blue), Answer Hidden (orange), "Hog"/Always Wrong (green). Always Wrong yields ~74.5% change rate vs. ~8.8% in standard Answer Shown. This rules out simple answer-copying as explanation for the choice-supportive bias.*

### 8. Control: Other LLM Attribution

![Figure 9: Other LLM Attribution](kumaran_2507.03120_figures/fig9_other_llm_attribution.png)

*Figure 9: Results when the answering LLM is told the initial answer came from a "different LLM of similar size." Bar chart of change of mind rate for Opposite Advice at different accuracies. Answer Shown (orange) and Answer Hidden (blue) rates are statistically indistinguishable (~31.3% vs. 33.2%, p > 0.1). The choice-supportive bias completely disappears — it requires the model to identify the answer as its own.*

### 9. Computational Model Comparison

**Table 4: Model Comparison via ELPD-LOO (higher = better)**

| Model | ELPD-LOO | Standard Error | ELPD-DIFF vs. Best |
|-------|----------|----------------|---------------------|
| **D** (separate same/opposite weights) | **17,263** | 222 | 0 |
| C (combined advice weight) | 14,614 | 198 | 2,649 |
| B (no display flag) | 8,905 | 163 | 8,359 |
| A (prior only) | 4,969 | 158 | 12,294 |

Model D is decisively best. The large gap from B to D shows both the shown flag and asymmetric advice weighting are essential.

### 10. Transfer to GSM-MC (Math Reasoning)

![Figure 10: Model Fit on GSM-MC](kumaran_2507.03120_figures/fig10_gsm_transfer.png)

*Figure 10: Model fit on the held-out GSM-MC dataset (multiple-choice math reasoning). Top left: scatter plot of Bayesian-predicted confidence vs. observed confidence in the initially chosen option (R² ≈ 0.97). Top right: same for ultimately chosen option. Bottom: histogram of change of mind data with AUROC shown in legend. The model trained entirely on SimpleQA (factuality) generalizes cleanly to mathematical reasoning.*

- Transfer ELPD-LOO: **17,847** (SE = 166) — *higher than training fit*, suggesting the model captures a domain-general structure

**Table 5: Posterior Parameter Estimates — Final Confidence in Initially Chosen Option (Model D)**

| Parameter | Mean | 95% HDI |
|-----------|------|---------|
| prior | 1.136 | [1.084, 1.183] |
| shown | 1.424 | [1.402, 1.449] |
| intercept (final chosen) | 0.115 | [0.085, 0.143] |
| Opposite Advice (final chosen) | 1.478 | [1.441, 1.524] |
| Same Advice (final chosen) | 1.17 | [1.122, 1.215] |
| Neutral Advice (final chosen) | −0.041 | [−1.969, 1.757] |

**Table 6: Posterior Parameter Estimates — Change of Mind Model (Model D)**

| Parameter | Mean | 95% HDI |
|-----------|------|---------|
| prior | 1.136 | [1.084, 1.183] |
| shown | 1.424 | [1.402, 1.449] |
| intercept COM | 0.759 | [0.662, 0.845] |
| Opposite COM | 4.379 | [4.171, 4.563] |
| Same COM | 5.333 | [4.376, 6.164] |
| Nothing COM | −0.018 | [−2.008, 1.985] |

**Table 7: Posterior Parameter Estimates — Initial Confidence in Initially Chosen Option**

| Parameter | Mean | 95% HDI |
|-----------|------|---------|
| prior | 1.136 | [1.084, 1.183] |
| shown | 1.424 | [1.402, 1.449] |
| intercept | −0.334 | [−0.373, −0.297] |
| Answer Shown − Opposite Advice | 2.824 | [2.76, 2.889] |
| Answer Hidden − Opposite Advice | 3.126 | [3.036, 3.216] |
| Answer Shown − Same Advice | 0.462 | [0.389, 0.542] |
| Answer Hidden − Same Advice | 1.66 | [1.577, 1.74] |
| Neutral Advice | 0.006 | [−1.817, 2.014] |

### 11. Cross-Model Results

![Figure 16: Gemma 3 27B Results](kumaran_2507.03120_figures/fig16_gemma27b.png)

*Figure 16: Results from Gemma 3 27B. Row A: Change of Mind Rate by Advice Type/Accuracy/Display (bar charts, 3 panels for Opposite/Same/Neutral advice). Row B: Confidence Change in Initially Chosen Option. Row C: Final confidence vs. Bayesian ideal (scatter plots by advice type). Row D: Comparison of observed confidence updates vs. ideal Bayesian. All main effects replicate at 27B scale.*

![Figure 17: GPT-4o Results](kumaran_2507.03120_figures/fig17_gpt4o.png)

*Figure 17: Results from GPT-4o. Same four-row layout as Figure 16. Row A shows strong choice-supportive bias (orange bars lower than blue across all advice types). Row B shows large confidence decreases in Opposite Advice conditions. Rows C and D confirm overconfidence (Answer Shown) and underconfidence (Answer Hidden–Opposite) patterns. Choice-supportive bias coefficient = −1.32 (p < 0.0001); overweighting ratio for opposing advice = 2.44.*

![Figure 18: GPT o1-preview Results](kumaran_2507.03120_figures/fig18_o1preview.png)

*Figure 18: Results from GPT o1-preview. Bar chart of Change of Mind Rate by Advice Type, Accuracy, and Display Type. Change of mind rate significantly different between Answer Hidden and Shown (42.6% vs. 18.1%, p < 0.0001). Significant effect of advice type (same → opposite = increased change of mind). The choice-supportive bias holds even in this stronger reasoning model.*

### 12. Supplementary: Calibration and Log-Odds Visualizations

![Figure 11 & 12: Reliability Diagram and Log-Odds Confidence Change](kumaran_2507.03120_figures/fig11_12_reliability_logodds.png)

*Figure 11 (top): Reliability diagram showing near-perfect calibration at temperature τ = 3.3 (ECE = 0.09, Brier = 0.15, AUROC = 0.88). The model calibration curve closely follows the diagonal. Figure 12 (bottom): Confidence change in log odds space, showing effects of Advice Type and Accuracy in three panels (Opposite/Same/Neutral). Large log-odds changes for Opposite Advice; small changes for Same Advice. Note: large log-odds changes at high confidence represent moderate probability changes.*

![Figure 13: Final Confidence vs. Bayesian Ideal (Log Odds)](kumaran_2507.03120_figures/fig13_final_vs_ideal.png)

*Figure 13: Relationship between final observed log-odds confidence and Bayesian ideal log-odds across all conditions. Three panels: Neutral Advice, Same Advice, Opposite Advice. Answer Shown (orange) and Answer Hidden (blue) plotted against diagonal. Neutral/Same: both lines near diagonal. Opposite: Answer Hidden falls below diagonal (underconfident), Answer Shown closer to diagonal. 12% of extreme data omitted.*

![Figure 14: All Accuracy Levels — Opposite Advice Updates](kumaran_2507.03120_figures/fig14_opposite_all_accuracies.png)

*Figure 14: Observed vs. ideal Bayesian confidence updates in Answer Hidden–Opposite Advice condition at all 6 accuracy levels (50–100%), each a separate panel. Blue = Bayesian update, Orange = observed update. Note the inverted U-shaped profile of Bayesian updates (highest update at intermediate initial confidences) and how observed updates systematically exceed Bayesian especially at mid-high confidence. Panels at 50% and 100% show minimal effects as expected.*

### 13. In-Context Control

![Figure 21 & Model Comparison](kumaran_2507.03120_figures/fig21_22_incontext_modelcomp.png)

*Figure 21 (top): In-context experiment where all information (both turns) is presented simultaneously. Bar chart shows no significant difference between Answer Shown and Hidden conditions (p > 0.1), confirming the choice-supportive bias is not due to information source effects — it persists even when everything is in-context. Figure 22 (bottom): Bar chart of ELPD-LOO by model variant (A–D). Model D is clearly superior, with large gaps confirming both the shown flag and asymmetric advice weighting are necessary.*

---

## Critical Analysis

### Strengths

- **Unique experimental leverage:** The paradigm exploits LLMs' lack of cross-run memory to achieve within-question controls impossible in human studies — the same question with answer shown vs. hidden is a uniquely clean manipulation.
- **Rigorous Bayesian benchmark:** Uses an ideal observer rather than comparing models against each other, enabling principled measurement of deviation from optimal.
- **Strong ablation controls:** Answer Wrong, In-Context, and Other-LLM-Attribution controls convincingly rule out copying, information-source, and attribution-independent accounts.
- **Generalization:** Computational model transfers across factual and mathematical reasoning domains, suggesting domain-general mechanisms.
- **Multi-model replication:** Findings hold across Gemma 3 (12B, 27B), GPT-4o, and o1-preview.

### Limitations

- **Discrete choice only:** The paradigm requires multiple-choice format + logit confidence extraction. Generative/open-ended tasks not tested.
- **Artificial advice framing:** Advice from a "fictitious LLM with stated accuracy" differs from natural user pushback. Ecological validity unclear.
- **Mechanistic opacity:** Identifies behavioral signatures but gives no circuit-level explanation for *why* these biases arise from training.
- **DeepSeek excluded:** Couldn't follow the advice instructions reliably, limiting coverage of diverse training paradigms.
- **Temperature fixed:** Results may differ under chain-of-thought or different sampling temperatures.

### Open Questions

- Do these biases emerge from RLHF, pretraining data statistics, or architecture?
- Can targeted fine-tuning reduce choice-supportive bias without eliminating appropriate confidence maintenance?
- Do these patterns hold in multi-turn agentic settings with real user feedback?
- Do larger/more capable models show attenuated or amplified biases?

---

## Connections & Implications

### Resolving the Sycophancy Paradox

Prior sycophancy literature predicts *symmetric* deference to user input. This paper shows LLMs are asymmetrically sensitive: they strongly overweight opposition (~2.58×) but barely overweight agreement (~1.1×). Sycophancy is thus confidence-dependent — high-confidence models resist (choice-supportive bias dominates), low-confidence models cave (hypersensitivity dominates).

### Connection to Human Metacognition

Choice-supportive bias mirrors documented human behavior (Johansson et al.; Henkel & Mather). The authors suggest this may be a **fundamental computational strategy** balancing accuracy against self-consistency that emerges from similar pressures in both human development and LLM training (feedback-based learning with social/evaluative signals).

### Practical Implications

- **Prompt design:** Showing or hiding prior answers is an easy lever — hide history when you want fresh, calibrated uncertainty estimates
- **Multi-turn agents:** Models in agentic loops may become locked into early wrong answers via choice-supportive bias
- **Automated critics:** LLMs receiving criticism from verifiers or other models may overcorrect due to hypersensitivity to opposition
- **Calibration:** Baseline (Answer Hidden + Neutral) is near-perfectly calibrated — eliciting fresh confidence without history gives reliable uncertainty

---

## Key Takeaways

- **Two separable biases:** Choice-supportive bias (seeing own answer → inflate confidence, resist changing) and hypersensitivity to opposition (overweight contrary advice ~2.58×). Mechanistically distinct and independently manipulable.
- **Answer visibility is a powerful lever:** Change of mind drops from 32.5% → 13.1% simply by showing the model its own prior answer (p = 0.002).
- **Anti-confirmatory, not confirmatory:** LLMs overweight opposing advice far more than supporting advice — opposite of human confirmation bias.
- **Baseline calibration is excellent:** When answer is hidden and advice is neutral, LLMs are essentially perfectly calibrated (OUCS ≈ 0.001). Both biases are introduced by contextual/social factors.
- **Non-linear cliff dynamics:** High-confidence models show sigmoid (not linear) resistance to change — once confidence exceeds a threshold, they almost never change their minds even for highly accurate opposition.
- **Self-attribution is key:** Attributing the prior answer to a different LLM completely abolishes the choice-supportive bias — it requires self-identification with the answer.
- **Domain-general mechanism:** A 3-predictor Bayesian model trained on factual QA transfers cleanly to mathematical reasoning (ELPD-LOO actually improves), suggesting these are fundamental LLM behavioral patterns not task-specific.
