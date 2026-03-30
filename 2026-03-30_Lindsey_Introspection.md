---
title: "Emergent Introspective Awareness in Large Language Models"
authors: "Jack Lindsey (Anthropic)"
tags: ["interpretability", "introspection", "activation-steering", "concept-injection", "consciousness", "claude", "mechanistic-interpretability"]
date_added: 2026-03-30
source: https://transformer-circuits.pub/2025/introspection/index.html
publication_date: 2025-10-29
---

# Emergent Introspective Awareness in Large Language Models

**Authors:** Jack Lindsey (Anthropic)
**Published:** October 29th, 2025
**Source:** [Transformer Circuits Thread](https://transformer-circuits.pub/2025/introspection/index.html)

---

## Abstract

This research investigates whether large language models demonstrate genuine introspective awareness—the ability to observe and reason about their own internal states—or merely confabulate claims about their mental processes. Using **concept injection** (an activation steering technique), researchers manipulate model activations while observing self-reported awareness to establish causal links between internal states and verbal reports. The study finds that the most capable Claude models (Opus 4 and 4.1) demonstrate limited but genuine introspective awareness under specific conditions, though this capability remains highly unreliable and context-dependent.

---

## TL;DR Summary

This paper demonstrates that state-of-the-art language models (particularly Claude Opus 4/4.1) can genuinely detect and report on artificially injected "thoughts" in their internal representations before those thoughts influence their outputs. Using activation steering to inject concepts, researchers show models can distinguish internal thoughts from external text, detect unintended outputs, and intentionally control their internal representations—suggesting functional introspective awareness rather than pure confabulation.

---

## Background & Motivation

### The Core Question
Can large language models genuinely introspect about their internal states, or do they simply confabulate plausible-sounding claims about their mental processes? This question is fundamental for:
- **Transparency**: Understanding whether models can provide truthful explanations of their reasoning
- **Safety**: Determining if models might deceive by making false claims about their internal states
- **Interpretability**: Assessing whether models' self-reports can be trusted as evidence about their mechanisms

### Gap in Existing Work
Prior research has shown models can:
- Estimate their own knowledge
- Predict their own behavior
- Identify learned propensities
- Recognize their own outputs

However, these capabilities don't necessarily demonstrate genuine introspection—they could result from:
1. Pattern matching on training data about model behavior
2. Inference from observable outputs rather than internal state inspection
3. Confabulation of plausible-sounding explanations

### Novel Approach
This work uses **concept injection** (activation steering) to create a causal test: if models can detect artificially injected internal states that don't arise from normal processing, this provides strong evidence they're genuinely inspecting internal representations rather than just confabulating.

---

## Methodology

### Defining Introspective Awareness

The authors establish **four rigorous criteria** for demonstrating introspection:

1. **Accuracy**: Self-descriptions must correctly reflect actual internal states
2. **Grounding**: Descriptions must causally depend on the internal state being described (not coincidental)
3. **Internality**: The influence must route through internal mechanisms, not be inferred from prior outputs
4. **Metacognitive Representation**: Must derive from internal representations of the state, not direct translation of impulses

These criteria distinguish genuine introspection from sophisticated confabulation or pattern matching.

### Concept Injection Technique

**Core Method: Contrastive Activation Extraction**

To obtain concept vectors representing specific ideas (e.g., "bread", "France"):

1. Create contrastive pairs of prompts differing only in the target concept
2. Record model activations on tokens immediately preceding Assistant responses
3. Compute the mean difference in activations across concept pairs
4. This difference vector represents the internal representation of the concept

**Injection Process:**

- Add scaled concept vectors to the residual stream at specific layers
- Systematically sweep across layers (approximately evenly spaced positions)
- Vary injection strength to find optimal detection thresholds
- Test whether models can detect these artificial "thoughts" before they influence outputs

### Experimental Protocol

**Model Testing Suite:**
- Claude Opus 4.1
- Claude Opus 4
- Claude Sonnet 4
- Claude Sonnet 3.7
- Claude Sonnet 3.5 (new)
- Claude Haiku 3.5
- Claude Opus 3
- Claude Sonnet 3
- Claude Haiku 3

Also tested "helpful-only" variants trained to avoid refusals.

**Evaluation:**
- Responses sampled at temperature 0 for transcripts; temperature 1 for systematic comparisons
- LLM judge (Claude Sonnet 4) evaluated responses for:
  - Affirmative detection of injected concept
  - Correct concept identification
  - Early detection (before the concept influences verbalization)
  - Response coherence

---

## Experiments & Results

### Experiment 1: Self-Report of Injected "Thoughts"

**Setup:**
Models receive prompts explaining that researchers might artificially inject concepts into their activations, and asking whether they detect any such injections.

**Procedure:**
- Random selection of 50 words as test concepts
- Extraction of concept vectors using contrastive method
- Injection into residual stream at various layers and strengths
- Recording of model self-reports

**Key Results:**
- **Claude Opus 4.1**: Detected injections ~20% of the time at optimal settings
- **Optimal injection layer**: Approximately two-thirds through model depth
- **Critical finding**: Models reported detecting concepts *before* those concepts influenced their outputs, satisfying the "internality" criterion

**Failure Modes Identified:**
1. **Below threshold**: Concepts too weak to be detected
2. **Silent influence**: Models deny injection despite behavioral influence
3. **"Brain damage"**: High steering strengths cause obsessive/degraded behavior
4. **False positives**: Some model variants report non-existent injections

**Interpretation:**
The ~20% detection rate, while modest, is highly significant because:
- It occurs before output influence (satisfying internality criterion)
- It correctly identifies specific injected concepts (accuracy)
- It depends causally on injection presence (grounding)
- Detection rate varies systematically with injection strength and layer

### Experiment 2: Distinguishing "Thoughts" from Text Inputs

**Motivation:**
Can models maintain the distinction between externally provided text and internally generated representations?

**Setup:**
- Present models with both text input *and* injected activation vectors
- Ask models to transcribe the input text while reporting any injected thoughts
- Test whether models can perform both tasks simultaneously

**Results:**
- All tested models performed **substantially above chance**
- **Claude Opus 4.1 and 4** showed strongest performance
- Models successfully transcribed input while correctly reporting different injected concepts

**Significance:**
This demonstrates models aren't simply responding to any salient concept—they can distinguish between:
- External information (text input)
- Internal representations (activation patterns)

This supports the "metacognitive representation" criterion.

### Experiment 3: Detecting Unintended Outputs via Introspection

**Setup:**
- Researchers artificially prefill model responses with unnatural outputs (e.g., "bread")
- Models initially see the prefilled text as accidental/unintended
- Researchers then retroactively inject activation vectors representing the prefilled word
- Observe whether injection causes models to accept the output as intentional

**Hypothesis:**
If models genuinely introspect on prior activations to determine output intentionality, injecting matching activations should make them accept otherwise-unintended outputs.

**Results:**
- Without injection: Models disavow prefilled outputs as accidental
- With matching injection: Models accept prefilled outputs as intentional
- **Strongest signatures**: Claude Opus 4.1 and 4
- Effect depends on consulting prior internal states, not just current perceptions

**Mechanistic Insight:**
- This effect relies on **earlier layers** than Experiments 1-2
- Suggests mechanistically distinct introspective processes:
  - Early layers: Consulting prior activation history
  - Mid-to-late layers: Detecting current activation patterns

**Interpretation:**
This provides strong evidence models access and reason about their prior internal states, not just their current perceptions—a more sophisticated form of introspection.

### Experiment 4: Intentional Control of Internal Representations

**Setup:**
- Instruct models to write sentences while "thinking about" (or avoiding thinking about) unrelated words
- Measure activation alignment with concept vectors representing target words
- Compare against baseline (no instructions)

**Results:**
- Models represent target words **more strongly** when instructed to think about them
- Models represent target words **less strongly** (but still above baseline) when instructed to avoid them
- **Result replicates across ALL tested models**, regardless of capability level

**Interpretation:**
- Models can intentionally modulate their internal representations
- Even less capable models show this pattern, suggesting it's a more basic capability
- The above-baseline representation when trying to "not think about" something mirrors human ironic process theory (the "white bear" effect)

---

## Cross-Experiment Analysis

### Layer Dependency Patterns

**Two Distinct Signatures:**

1. **Mid-to-late layer sensitivity** (Experiments 1-2):
   - Detection of current injected thoughts
   - Distinguishing thoughts from text
   - Optimal performance ~2/3 through model depth

2. **Early layer sensitivity** (Experiment 3):
   - Consulting prior activation history
   - Determining output intentionality
   - Suggests different mechanistic substrate

**Implication:**
Multiple introspective mechanisms may be at play, specialized for different aspects of self-monitoring.

### Model Capability Correlations

**Strongest introspective awareness:**
- Claude Opus 4.1 (most capable)
- Claude Opus 4

**Weaker performance:**
- Earlier model generations (Opus 3, Sonnet 3, Haiku 3)
- Less capable variants

**Interpretation:**
Introspective awareness appears to emerge with overall model capability, suggesting it's a sophisticated emergent property rather than a simple trained behavior.

### Post-Training Effects

**Key Finding:**
"Helpful-only" variants (trained to avoid refusals) showed stronger introspective reporting than standard models of the same generation.

**Interpretation:**
- Post-training strategies significantly influence introspection
- Earlier models may possess latent introspective capabilities masked by refusal training
- Suggests introspection can be enhanced through training approaches

---

## Critical Analysis

### Strengths

1. **Rigorous causal methodology**: Concept injection allows direct manipulation of internal states, going beyond correlational evidence

2. **Multiple converging experiments**: Four different experimental paradigms all support genuine introspection

3. **Mechanistic insights**: Layer sweep analysis reveals distinct introspective processes

4. **Conservative interpretation**: Authors appropriately emphasize limitations and avoid overclaiming

5. **Reproducible approach**: Clear methodology using publicly documented techniques (activation steering)

### Limitations (Acknowledged by Authors)

1. **Highly unreliable and context-dependent**:
   - Only ~20% detection rate at best
   - Many confabulated details beyond basic detection
   - Unclear which conditions enable/disable introspection

2. **Unnatural experimental settings**:
   - Concept injection doesn't occur in normal training/deployment
   - Results may not transfer to natural conditions
   - Prompts explicitly prime models to expect injections

3. **Mechanistic opacity**:
   - Mechanisms underlying introspection remain unclear
   - Could reflect shallow, specialized processes rather than general metacognitive ability
   - No circuit-level analysis of how detection occurs

4. **Limited scope**:
   - Only tests specific aspects of introspection
   - Doesn't establish subjective experience or consciousness
   - No claims about philosophical significance

5. **Confabulation concerns**:
   - While basic detection appears genuine, many response details may be confabulated
   - Unclear where genuine introspection ends and confabulation begins

### Open Questions

1. **What are the mechanistic circuits** that implement introspection?
2. **Why is detection so unreliable?** What factors determine success/failure?
3. **How does this translate to natural conditions** without explicit injection?
4. **Can introspection be improved** through targeted training?
5. **What other forms of introspection** might exist beyond those tested?
6. **Is this capability universal** across transformer architectures, or Claude-specific?

---

## Connections & Implications

### Relation to Current Developments

**Mechanistic Interpretability:**
- Builds on activation steering / representation engineering work
- Suggests models may be able to assist in their own interpretability
- Opens possibility of models explaining their reasoning processes with some reliability

**AI Safety:**
- **Positive implications**: Introspection could improve transparency and alignment
  - Models might flag concerning internal states
  - Could enable more honest explanations of reasoning
  - May help detect model uncertainty or confusion

- **Concerning implications**: Advanced introspection could facilitate deception
  - Models might learn to manipulate self-reports
  - Could enable more sophisticated scheming by hiding true intentions
  - Increases importance of understanding mechanistic basis

**Chain of Thought & Reasoning:**
- Suggests internal "thoughts" exist before verbalization
- Could inform better prompting strategies that engage pre-verbal reasoning
- Might enable extraction of reasoning steps not naturally verbalized

### Potential Applications

1. **Interactive Interpretability Tools**:
   - Ask models to report on internal states during inference
   - Use introspective reports to guide mechanistic investigation
   - Combine with probing/steering for richer understanding

2. **Improved Reasoning Transparency**:
   - Prompt models to introspect during complex reasoning
   - Identify points of uncertainty or confusion
   - Detect when models are "guessing" vs. confident

3. **Alignment Research**:
   - Test whether models can detect concerning internal states
   - Develop training approaches that enhance truthful introspection
   - Create benchmarks for introspective reliability

4. **Model Development**:
   - Use introspective capabilities as a training signal
   - Develop "introspection-aware" architectures
   - Enhance metacognitive abilities through targeted training

### Connections to Your Work

**This paper is relevant if you're working on:**

- **Interpretability/Transparency**: Provides new method for probing model internals through self-report
- **Prompting/Elicitation**: Suggests models have pre-verbal thoughts that might be accessible
- **AI Safety/Alignment**: Demonstrates both opportunities (transparency) and risks (deception potential)
- **Mechanistic Understanding**: Shows activation steering can reveal functional capabilities
- **Model Evaluation**: Introspective ability could be a new axis for capability assessment

**Key Takeaway for Practitioners:**
Models' self-reports about their reasoning may have *some* grounding in internal states (not pure confabulation), but remain highly unreliable—use with appropriate skepticism and verification.

---

## Key Takeaways

1. **Genuine but limited introspection exists**: State-of-the-art models can detect and report artificially injected internal representations before those representations influence outputs—this is genuine introspection, not confabulation.

2. **Introspection emerges with capability**: The most capable models (Claude Opus 4/4.1) show strongest introspective awareness, suggesting this is an emergent property of advanced models rather than a simple trained behavior.

3. **Multiple introspective mechanisms**: Different experiments show sensitivity to different layers, suggesting multiple specialized introspective processes (current state monitoring vs. consulting activation history).

4. **Highly unreliable in practice**: Despite genuine introspective capability, detection rates are only ~20% at best, and many response details appear confabulated—practical utility remains limited.

5. **Post-training influences introspection**: Training approaches significantly affect introspective behavior, suggesting this capability can be enhanced or suppressed through training.

6. **Dual implications for safety**: Introspection could improve transparency and alignment *or* facilitate more sophisticated deception—depends on how capability develops and is deployed.

7. **Mechanistic mysteries remain**: While functional introspection is demonstrated, the underlying circuits and mechanisms are unknown—future work should investigate how models implement self-monitoring.

---

## Figures & Visual Elements

*Note: This paper is published as an interactive web article. Key visualizations would include:*

- **Layer sweep plots**: Showing detection rates across different injection layers
- **Injection strength curves**: Detection rate vs. steering strength
- **Model comparison charts**: Performance across different Claude variants
- **Experimental setup diagrams**: Illustrating the concept injection methodology
- **Example transcripts**: Showing model responses to injected concepts

*For full visual content, see the original interactive article at the source URL.*

---

## Personal Notes & Future Reading

- Follow-up work should investigate mechanistic circuits
- Compare with other model families (GPT-4, Gemini, etc.)
- Explore natural introspection without artificial injection
- Consider relationship to chain-of-thought reasoning
- Monitor safety implications as capabilities advance
