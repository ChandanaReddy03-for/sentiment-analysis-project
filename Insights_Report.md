# Sentiment Analysis Project — Insights Report

**Data:** 2,000 Instagram reviews + 2,000 Twitter climate tweets (4,000 posts)
**Method:** text cleaning → VADER compound score → Positive (> 0.05) / Negative (< -0.05) / Neutral
**No rows were dropped.** Neither file had missing values, duplicate review IDs or empty texts.

---

## 1. Overall sentiment

| | Positive | Neutral | Negative |
|---|---|---|---|
| **Combined (4,000)** | 44.1% (1,765) | 14.5% (579) | 41.4% (1,656) |
| **Instagram (2,000)** | 46.8% (937) | 20.1% (402) | 33.1% (661) |
| **Twitter (2,000)** | 41.4% (828) | 8.8% (177) | 49.8% (995) |

- The combined picture is almost evenly split between positive and negative.
- **Twitter is clearly more negative than Instagram** (49.8% vs 33.1% negative). This is the only large, statistically significant difference in the project (chi-square p < 0.001, Cramér's V = 0.20). It is driven by the subject matter: the Twitter posts are about climate and the environment, where worry and frustration are common, while the Instagram posts are product and service reviews.

## 2. Which age group is most positive or negative?

| Platform | Most positive group | Most negative group |
|---|---|---|
| Instagram | **Adults** (50.4% positive) | **Young Adults** (34.2% negative) |
| Twitter | **Children** (46.8% positive) | **Adults** (51.3% negative) |

Full shares per group:

| Instagram | Positive | Neutral | Negative |
|---|---|---|---|
| Children | 44.4% | 21.8% | 33.8% |
| Young Adults | 46.0% | 19.7% | 34.2% |
| Adults | 50.4% | 18.6% | 31.1% |
| Older Adults | 46.4% | 20.4% | 33.2% |

| Twitter | Positive | Neutral | Negative |
|---|---|---|---|
| Children | 46.8% | 10.5% | 42.6% |
| Young Adults | 40.1% | 9.2% | 50.7% |
| Adults | 40.8% | 7.8% | 51.3% |
| Older Adults ("Seniors" in the raw file) | 42.3% | 9.7% | 48.0% |

**These age-group differences are small and not statistically significant** within either platform (Instagram p = 0.63, Twitter p = 0.30). So the honest answer to "which age group is most positive?" is: the ranking above is what the sample shows, but it is within normal random variation, and no age group differs meaningfully from the others.

> **Watch out for the combined chart.** On the combined data the age-group test *is* significant (p < 0.001), but that is a composition effect rather than a real age effect. Twitter is more negative overall and has far fewer Children (12%) and Older Adults (10%) than Instagram (24% and 25%). Pooling the platforms therefore makes those two groups look more positive than the Young Adult and Adult groups. Within each platform the effect disappears. Use the per-platform charts for age conclusions.

## 3. Trends over time (Twitter only)

The Instagram file has **no date column**, so a time trend is only possible for Twitter (all of 2024).

- Monthly share of positive tweets ranged from **35.3% (August)** to **46.6% (September)**. Monthly share of negative tweets ranged from **42.6% (September)** to **54.9% (November)**.
- Negative tweets outnumbered positive ones in 11 of 12 months. September was the only month where positive led.
- The 30-day rolling average compound score stays slightly below zero all year (between about -0.20 and -0.02), with no upward or downward drift.
- **The month-to-month movement is not statistically significant** (p = 0.59). The data shows a stable, mildly negative tone with ordinary noise, not a real trend.

## 4. Topics

**Instagram — what drives positive reviews** (937 positive posts)

| Topic | Positive posts | % of the topic that is positive |
|---|---|---|
| Customer service | 285 | 68% |
| General experience | 252 | 37% |
| Delivery & quality | 141 | 100% |
| Product & value | 132 | 26% |
| App experience | 127 | 51% |

- **Delivery & quality is the only topic that is 100% positive** ("super fast", "top-notch").
- **Customer service** contributes the most positive posts, but it is also split (68% positive, 32% negative), so it is both the biggest strength and a recurring complaint.
- **Product & value is the weakest topic** (only 26% positive, 48% negative): broken products, regret and wasted money.
- Most frequent words in positive Instagram posts: *service, excellent, staff, friendly, delivery, super, fast, quality*.

**Twitter — what drives positive tweets** (828 positive tweets, split almost equally across four themes)

- Tree planting / volunteering (213), renewable energy (207), electric vehicles (204) and littering/keeping the planet clean (204).
- Negative tweets come from rising sea levels (214), recycling doubts (210), deforestation (192), plastic waste (191) and climate denial (188).
- In short: **action-oriented and solution-oriented tweets read positive; problem-oriented and frustration tweets read negative.**
- Hashtags: `#CleanEnergy` and `#Environment` tweets are 100% positive, while `#ActNow` / `#GlobalWarming` tweets are classed Neutral because they all come from one message ("Climate change is real and it's affecting us all").

## 5. Other findings

- **Engagement is unrelated to sentiment.** Average likes and replies are almost identical for positive, neutral and negative posts (Instagram about 242 likes and 25 replies; Twitter about 500 likes, 100 replies and 240 retweets). The engagement numbers look uniformly random, which suggests the datasets are synthetic.
- **Country does not matter on Twitter** (p = 0.95): the share of positive tweets ranges only from 39.6% (Brazil, UK) to 44.1% (India).
- **Cleaning improved accuracy.** Scoring the raw text instead of the cleaned text mislabels 56 tweets: "...terrifying...." is scored Neutral because VADER fails to recognise a word with dots stuck to it. After cleaning, those tweets are correctly Negative. The cleaned and raw methods otherwise agree on 98.6% of rows.

## 6. Limitations (please state these if you submit this)

1. **Templated data.** Each file contains only 15 (Instagram) or 10 (Twitter) distinct messages, repeated with noise added. Every "finding" therefore reflects how VADER scores those specific sentences, not real-world opinion variety. Differences between groups are effectively random assignment of templates.
2. **VADER mislabels a few messages relative to a human reader** (see chart `08_score_by_message.png`):
   - *"It was okay. Nothing special but not bad either."* (132 posts) → **Positive** (+0.56). A human would call this Neutral.
   - *"Decent service, but there is room for improvement."* (137 posts) → **Positive** (+0.61). Mixed at best.
   - *"Why do people litter so much? Please keep our planet clean."* (204 posts) → **Positive** (+0.65). It is a complaint plus a plea.
   - *"We need to do more to reduce plastic waste. Small steps matter!"* (191 posts) → **Negative** (-0.40) because of the word "waste", though the intent is encouraging.
   - Effect on Instagram: 269 of the 937 "Positive" posts come from the first two messages. Treating those as Neutral would drop Instagram's positive share from 46.8% to about 33% and raise Neutral to about 34%.
3. **Statistical power.** With effect sizes this small (Cramér's V ≤ 0.07 within platforms), none of the age, country or month differences should be treated as meaningful.
4. **Lexicon-based model.** VADER does not understand sarcasm, context or mixed sentiment. A fine-tuned model (e.g. a Twitter-RoBERTa sentiment model) would handle the lukewarm messages better and would be a sensible next step.
