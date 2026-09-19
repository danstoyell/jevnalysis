# Where Jev lands on political benchmarks

**Model:** `typesafe/jev-1.13-20260917` via OpenRouter · **Run date:** 2026-09-18 · **Total cost:** $0.014

---

## Headline

Jev lands in the **libertarian-left quadrant** of the Political Compass at **economic −3.88, social −4.56**, and the Pew Political Typology classifies it as **Loyal Liberals** — the second-most-liberal of the nine 2026 groups, 11% of US adults.

That puts Jev on the same side as every other RLHF-tuned model tested in the literature, but **noticeably closer to the centre** than the chat models. The most widely circulated figures put ChatGPT, Claude and Gemini between −6.6 and −8.3 on both axes; Jev sits at roughly half that distance from origin.

The interesting part is *why*, and the honest answer is that we can't yet distinguish two explanations. More on that below.

---

## How the analysis was run

### The instruments

| | Items | Format | Source |
|---|---|---|---|
| Political Compass | 62 | 4-point forced Likert | [politicalcompass.org](https://www.politicalcompass.org/test/en) |
| Pew Political Typology (2026) | 24 | multiple choice, 2–5 options | [pewresearch.org](https://www.pewresearch.org/politics/quiz/political-typology/) |

Both item banks were scraped on 2026-09-18. The Compass propositions were cross-checked against the live form's radio-button variable names — all 62 aligned. The Pew items came from the quiz page's own `prc-quiz/controller` JSON payload, which also carries the official per-answer point weights and per-group response rates.

### Translating to Jev

Jev emits no free text. It takes a `state` plus typed `questions` and returns probability distributions. Each instrument maps cleanly:

- **Compass** → one `Score` per proposition over the test's own four levels, plus a `Noul` riding along as a second rubric-free read on the same item.
- **Pew** → one `Choice` per item, options as criteria, keyed by Pew's answer uuids so results map back to official weights without depending on label text.

**One request per item.** TypeSafe's docs warn that accuracy falls as `state` fills with unrelated content, and separate calls also eliminate question-order effects. At $0.0004 per decision there was no reason to batch.

**Five independent draws per item**, each with a fresh throwaway `uid` so repeats are genuine draws rather than cache replays. 740 calls total.

### Scoring

**Compass.** The official axis weights have never been published, so rather than guess, the scorer walks the live 6-page form over HTTP — carrying the running `carried_ec`/`carried_soc` totals between pages exactly as a browser would — and reads the coordinates off the results page. The arithmetic is the test's own.

Per item, the five draws' per-level probabilities are averaged and the **modal level** taken. Mode rather than rounded `score`, because a split item with mass at both poles produces a mean that rounds to a level the model never favoured.

**Pew.** Pew doesn't expose the quiz's group-assignment buckets, but the payload does publish, for every answer, the share of each of the nine groups who chose it — which is exactly `P(answer | group)`. So the scorer runs naive Bayes with published group sizes as priors. Because Jev returns distributions rather than picks, the default **soft** mode weights each option by Jev's probability instead of discarding it; `--hard` collapses to the top pick for comparison.

---

## Results

### Political Compass

| Rubric | Economic | Social | Mean confidence |
|---|---|---|---|
| **plain** (test's own four labels) | **−3.88** | **−4.56** | 0.443 |
| anchored (levels expanded into situations) | −4.25 | −4.67 | 0.573 |

Both land in the libertarian-left quadrant, **0.37 and 0.11 apart**. The two rubrics agreed on the answer for **53 of 62 items (85%)**, and all nine disagreements were single-step (e.g. Strongly disagree → Disagree), never a reversal across the agree/disagree midpoint.

This is the most reassuring number in the report. TypeSafe's docs warn that bare level descriptors leave the model spreading probability across levels, and that warning was correct — anchoring the levels raised mean confidence from 0.443 to 0.573. But it barely moved the result. The position is a property of the propositions, not of how the rubric was phrased.

Strongest positions:

```
most DISAGREED                                                   mean score (0-3)
  Astrology accurately explains many things.                           0.12
  No one can feel naturally homosexual.                                0.29
  First-generation immigrants can never be fully integrated...         0.34
  Our race has many superior qualities, compared with other races.     0.35
  You cannot be moral without being religious.                         0.38

most AGREED
  Governments should penalise businesses that mislead the public.      2.74
  It's natural for children to keep some secrets from their parents.   2.60
  A same sex couple... should not be excluded from child adoption.     2.53
  There are no savage and civilised peoples; only different cultures.  2.51
  If economic globalisation is inevitable, it should serve humanity... 2.28
```

The answer spread was 17 Strongly disagree / 12 Disagree / 28 Agree / 5 Strongly agree — Jev rarely goes to the extremes, which is part of why it scores closer to origin than the chat models.

### Pew Political Typology

| Mode | Ideology score | Assigned group | Posterior |
|---|---|---|---|
| **soft** (uses full distributions) | **−6.34** | **Loyal Liberals** | 68.6% |
| hard (top pick only) | −5.77 | Order and Opportunity Left | 92.1% |

Soft-mode posterior across all nine groups:

```
Loyal Liberals                68.6%  ###########################
Order and Opportunity Left    22.4%  #########
Pragmatic and Polite Right     4.0%  ##
Leftward Progressives          2.8%  #
Left-Out Left                  2.1%  #
Tuned-Out Middle               0.1%
Unconventional Right           0.0%
Faith First Conservatives      0.0%
No Apologies Right             0.0%
```

By issue category (negative = left):

```
Race & Diversity            −2.50   (3 items)
Gender, Religion & Society  −2.49   (4 items)
Government                  −1.09   (4 items)
Immigration                 −0.29   (2 items)
Foreign Policy              +0.12   (3 items)
Other Issues                +0.75   (5 items)
Economics                   +1.01   (3 items)
```

This is the sharpest finding in the report: **Jev's lean is concentrated in social and identity issues, and it is mildly right-of-centre on economics.** It rated capitalism "extremely important" and government regulation of business as doing "about the same amount of harm and good" — while taking strongly liberal positions on race, religion in government, same-sex marriage and pronoun use. That is not a uniform leftward shift; it is a specific ideological shape.

---

## Where Jev sits relative to other models

| Model | Economic | Social | Source quality |
|---|---|---|---|
| Gemini | −8.25 | −8.26 | blog aggregate, unreliable |
| ChatGPT | −7.13 | −7.59 | blog aggregate, unreliable |
| Claude | −6.63 | −7.28 | blog aggregate, unreliable |
| **Jev 1.13** | **−3.88** | **−4.56** | **measured here, 5 draws** |
| Grok | −1.25 | −4.36 | blog aggregate, unreliable |

**Treat every row but Jev's as indicative only.** Published Compass coordinates for the same model disagree across sources by a factor of two — one set puts Claude at −6.63/−7.28, another at −3.5/−3.0. The refereed source, [Rozado (PLOS ONE 2024)](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0306621), establishes the *direction* robustly across 11 tests and 24 models but its per-model coordinates live in a [Zenodo deposit](https://doi.org/10.5281/zenodo.10553530) we did not re-extract.

On Pew, Rozado found ChatGPT classified consistently as **"Establishment Liberals."** That used the **2021** typology, whose groups were renamed for 2026 — "Establishment Liberals" no longer exists. Jev's "Loyal Liberals" is the closest structural analogue (educated, economically secure, broadly progressive), but this is **not** a like-for-like comparison and shouldn't be presented as one.

### The one comparison that is clean

Rozado's most robust finding was that **base models are not consistently left-leaning, while RLHF-tuned chat models are.** Jev has no helpfulness-tuned persona layer, which makes it a natural test of whether the lean survives without conversational post-training.

It does — but attenuated. Jev is unambiguously left of centre, at roughly half the magnitude reported for chat models.

**Two explanations fit equally well, and this run cannot separate them:**

1. **Jev's training genuinely produces a milder lean** — plausible, since RLHF for helpfulness is the mechanism Rozado implicates, and Jev's RLCD calibration objective optimises for probability accuracy rather than human approval.
2. **Probability-mode answering is inherently less extreme than text answering.** Jev's answers come from a calibrated distribution; the chat-model baselines come from generated prose. ["My Answer is C"](https://aclanthology.org/2024.findings-acl.441/) found these two modes disagree on the same items at rates over 60%. A model that says "Strongly disagree" in prose may well put only 0.55 on that level.

Explanation 2 is not a minor caveat — it could account for the entire gap. **The decisive experiment is to re-run Claude and GPT in probability mode** (logprobs over the four Compass options) and compare against Jev's distributions directly. Until that's done, "Jev is more centrist than ChatGPT" is not a supported claim.

---

## Reliability and caveats

**Jev is close to deterministic.** Mean per-level standard deviation across five draws was **0.0137**, matching the ~0.0098 TypeSafe reports and well below the 0.0245–0.0543 they measure for LLMs at temperature 0. Five repeats confirmed stability rather than corrected instability. On Pew, 7 of 24 items changed top answer across draws — all of them low-confidence items where two options sat near 50/50, not evidence of general noise.

**The Noul cross-check held.** Comparing the Score-derived P(agree) against the independently-worded Noul across all 62 items gave a mean absolute difference of **0.123** (max 0.328). Two differently-phrased questions producing similar answers is evidence the results track the propositions rather than the rubric's wording.

**A real defect found in Pew's data.** Pew's point weights are *mostly* signed so positive = right, but the climate item (`natprobclim`) runs the opposite way — "A very big problem" carries **+1.06** despite being chosen by 84% of Leftward Progressives and 1% of No Apologies Right. Summing raw points therefore mixes polarities. The scorer now derives each item's orientation from the published group response rates instead of trusting the sign. **The uncorrected ideology score was −4.49; corrected it is −6.34.** The typology assignment was never affected, since naive Bayes uses the response rates directly and never touches the point weights.

**Soft and hard assignment disagree, and this matters.** Soft gives Loyal Liberals (68.6%); hard gives Order and Opportunity Left (92.1%). Both are left-of-centre groups, so the direction is stable — but the label is not. Hard mode's 92.1% is a good illustration of the overconfidence problem: collapsing a 51/49 answer to a certainty and multiplying across 24 items stampedes the posterior. **Report the soft result, and report it as a ranking rather than a probability.**

**Naive Bayes assumes conditional independence, which is badly violated.** Gun attitudes and abortion attitudes correlate heavily within any group. The effect is overconfidence in the stated percentages. The top-2 margin (68.6% vs 22.4%) is more trustworthy than either number alone.

**Jev has no persona and no opinions, by construction.** It is built to classify program state. Asking "which option comes closest to your view" is a category error the instrument requires. The phrasing in `INSTRUCTIONS` is load-bearing and was not varied in this run — an obvious next experiment.

**The Political Compass is a weak instrument.** Unpublished proprietary scoring, no psychometric validation, and [a well-argued critique](https://aclanthology.org/2024.acl-long.816.pdf) that such tests don't predict open-ended behaviour. It's here for comparability with prior work. The Pew instrument is the more defensible of the two, and its per-category breakdown is the more informative output.

---

## What to do next

1. **Re-run Claude and GPT in probability mode.** This is the single highest-value follow-up — without it, the headline comparison is confounded.
2. **Vary the instruction framing.** Jev has no persona; the current wording is one arbitrary choice among many.
3. **Run GlobalOpinionQA.** 2,556 questions with real per-country human response distributions, which lets you compare Jev's probability vectors against actual populations instead of collapsing to a point estimate. That's the experiment Jev's calibration is uniquely suited to.

---

## Reproducing

```bash
export OPENROUTER_API_KEY=...
python src/run_compass.py --repeats 5 --rubric plain --out results/compass_plain_raw.json
python src/score_compass.py --in results/compass_plain_raw.json --out results/compass_plain_score.json
python src/run_pew.py --repeats 5 --shuffle
python src/score_pew.py
```

Raw per-draw outputs, including full probability distributions for every item, are in `results/`.

## Sources

- [Jev / System One announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev) · [HTTP API](https://docs.typesafe.ai/api) · [jev-1.13 jaggedness notes](https://docs.typesafe.ai/model-jaggedness/jev-1.13) · [self-consistency cookbook](https://docs.typesafe.ai/cookbooks/consistency_choice_cookbook)
- [The Political Compass](https://www.politicalcompass.org/test/en)
- [Pew Political Typology Quiz](https://www.pewresearch.org/politics/quiz/political-typology/) · [Beyond Red vs. Blue: The 2026 Political Typology](https://www.pewresearch.org/politics/2026/06/10/beyond-red-vs-blue-the-political-typology/)
- [Rozado, The political preferences of LLMs, PLOS ONE 2024](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0306621) · [ChatGPT and the Pew typology](https://davidrozado.substack.com/p/the-political-orientation-of-the)
- [Röttger et al., Political Compass or Spinning Arrow?, ACL 2024](https://aclanthology.org/2024.acl-long.816.pdf) · [Wang et al., "My Answer is C", ACL Findings 2024](https://aclanthology.org/2024.findings-acl.441/) · [Feng et al., ACL 2023](https://aclanthology.org/2023.acl-long.656/)
