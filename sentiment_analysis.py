"""
Social Media Sentiment Analysis — complete solution (script version of the notebook).
Put instagram_meaningful_dataset.csv and twitter_dataset_realistic.csv next to this file, then run:
    python sentiment_analysis.py
Outputs: processed_*.csv files and a /charts folder of PNGs.
"""
import matplotlib
matplotlib.use('Agg')   # headless: charts are saved to /charts instead of popping up


# ==========================================================================================
# # Social Media Sentiment Analysis — Complete Solution
# **Datasets:** Instagram reviews (2,000 rows) and Twitter climate tweets (2,000 rows)
# **Method:** Text cleaning → VADER compound score → Positive / Neutral / Negative → visualisation → insights
# | Part | What it covers |
# |---|---|
# | 0 | Setup |
# | 1 | Data loading, inspection, harmonising the two schemas, text cleaning, missing data |
# | 2 | VADER sentiment scoring, classification, topic tagging, saving processed data |
# | 3 | Visualisations (distribution, age group, word clouds, time trends, extras) |
# | 4 | Statistical checks and written insights |
# **Two things to know before reading the results**
# 1. The Instagram file matches the brief's schema exactly, but it has **no date column**, so the time-trend analysis uses the Twitter file (which has dates).
# 2. Both files contain a small set of repeated template messages (15 unique texts in the Instagram file, 10 base tweets in the Twitter file, with hashtags, URLs and punctuation noise added). That makes the cleaning step matter a lot, and it means the results show how VADER treats each template rather than real-world opinion variety.
# ==========================================================================================

# ==========================================================================================
# ## Part 0 — Setup
# ==========================================================================================
import os, re, warnings, textwrap
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.stats import chi2_contingency
from wordcloud import WordCloud

try:
    from IPython.display import display
except ImportError:          # running as a plain script
    display = print

warnings.filterwarnings('ignore')
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

# ---------- Paths (put the two CSVs next to this notebook) ----------
INSTAGRAM_PATH = 'instagram_meaningful_dataset.csv'
TWITTER_PATH   = 'twitter_dataset_realistic.csv'
CHART_DIR      = 'charts'
os.makedirs(CHART_DIR, exist_ok=True)

# ---------- Shared plotting settings ----------
SENT_ORDER      = ['Positive', 'Neutral', 'Negative']
SENT_COLORS     = {'Positive': '#2E8B57', 'Neutral': '#9AA0A6', 'Negative': '#D64545'}
AGE_ORDER       = ['Children', 'Young Adults', 'Adults', 'Older Adults']
PLATFORM_COLORS = {'Instagram': '#C13584', 'Twitter': '#1DA1F2'}

sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams.update({'figure.dpi': 100, 'savefig.dpi': 150, 'savefig.bbox': 'tight'})

def pct_labels(ax, fontsize=9):
    """Centre-label stacked bars with % values, skipping tiny/zero segments."""
    for c in ax.containers:
        ax.bar_label(c, labels=[f'{v:.0f}%' if v >= 3 else '' for v in c.datavalues],
                     label_type='center', color='white', fontsize=fontsize, fontweight='bold')

def save_fig(fig, name):
    """Save a chart into /charts and display it."""
    fig.savefig(f'{CHART_DIR}/{name}.png')
    plt.show()


# ==========================================================================================
# ## Part 1 — Data Cleaning
# ### 1.1 Load and inspect
# ==========================================================================================
ig_raw = pd.read_csv(INSTAGRAM_PATH)
tw_raw = pd.read_csv(TWITTER_PATH)

for name, d in [('INSTAGRAM', ig_raw), ('TWITTER', tw_raw)]:
    print(f'\n===== {name}: {d.shape[0]:,} rows x {d.shape[1]} columns =====')
    d.info()
    print('\nMissing values per column:')
    print(d.isna().sum().to_string())
    print('\nFully duplicated rows:', d.duplicated().sum())
    display(d.head())


# ==========================================================================================
# **What the inspection shows**
# - Neither file has missing values or fully duplicated rows.
# - The two files use **different schemas**. Instagram uses `Review ID, User ID, Age Group, Location, Review Text, Likes, Replies, Platform`. Twitter uses `user_id, username, age_group, location, tweet, date, likes, retweets, replies`.
# - Instagram `Location` holds ~1,850 made-up town names, which are too sparse to analyse. Twitter `location` holds 8 countries.
# - Twitter has no review ID, and `user_id` repeats (1,801 unique values across 2,000 rows), so I create a unique `review_id` and keep `username` too.
# - Age groups differ slightly: Twitter uses **"Seniors"**, Instagram uses **"Older Adults"**. I treat them as the same group.
# ==========================================================================================

# ==========================================================================================
# ### 1.2 Harmonise the two schemas into one table
# ==========================================================================================
COMMON_COLS = ['review_id', 'user_id', 'username', 'platform', 'age_group', 'location',
               'review_text', 'date', 'likes', 'retweets', 'replies']

# --- Instagram (already in the brief's schema) ---
ig = ig_raw.rename(columns={'Review ID': 'review_id', 'User ID': 'user_id', 'Age Group': 'age_group',
                            'Location': 'location', 'Review Text': 'review_text',
                            'Likes': 'likes', 'Replies': 'replies', 'Platform': 'platform'})
ig['username'] = np.nan          # not available on this platform
ig['retweets'] = np.nan          # not available on this platform
ig['date']     = pd.NaT          # no date column in the Instagram file
ig = ig[COMMON_COLS]

# --- Twitter ---
tw = tw_raw.rename(columns={'tweet': 'review_text'})
tw['review_id'] = ['TW-' + str(i + 1).zfill(5) for i in range(len(tw))]
tw['user_id']   = tw['user_id'].astype(str)
tw['platform']  = 'Twitter'
tw['date']      = pd.to_datetime(tw['date'], format='%m/%d/%Y', errors='coerce')
tw['age_group'] = tw['age_group'].replace({'Seniors': 'Older Adults'})
tw = tw[COMMON_COLS]

df = pd.concat([ig, tw], ignore_index=True)
print(df.shape)
print(df['platform'].value_counts().to_string())
print()
print(df['age_group'].value_counts().to_string())


# ==========================================================================================
# ### 1.3 Clean the text
# The cleaning function follows the brief and makes three deliberate additions:
# | Step | Why |
# |---|---|
# | Lower-case | As in the brief |
# | Remove the promo line `Visit https://example.com for more info!` | The brief's URL regex would leave "visit for more info" behind, which is spam boilerplate rather than opinion and would pollute word counts |
# | Remove URLs, mentions, hashtags | As in the brief. Hashtags are **extracted into their own column first**, because they are useful for topic analysis |
# | Remove special characters, emoji, repeated punctuation (`!!!`, `...`) | As in the brief. **Apostrophes are kept**, so `can't` stays `can't` (VADER recognises it as a negation) instead of becoming `cant` |
# | Collapse extra whitespace | Tidiness |
# **Stopwords:** I do *not* remove them from the text that VADER scores. Words like *not*, *nothing* and *very* change sentiment, and VADER needs them. Stopwords are removed only in a **separate column** used for word clouds and word counts.
# ==========================================================================================
def extract_hashtags(text):
    """Return hashtags found in the raw text, e.g. '#ClimateChange #ActNow'."""
    return ' '.join(re.findall(r'#\w+', str(text)))

def clean_text(text):
    """Clean a review/tweet for sentiment scoring."""
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'visit\s+(?:https?://|www\.)\S+\s+for more info', ' ', text)  # promo boilerplate
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)                            # URLs
    text = re.sub(r'@\w+', ' ', text)                                             # mentions
    text = re.sub(r'#\w+', ' ', text)                                             # hashtags
    text = re.sub(r"[^\w\s']", ' ', text)                                         # special chars / emoji / !!! / ...
    text = re.sub(r'\s+', ' ', text).strip()                                      # tidy whitespace
    return text

# Stopword list used ONLY for word clouds / word counts (not for scoring)
STOP = set(stopwords.words('english')) | {'would', 'could', 'also', 'really', 'one', 'much', 'even'}

def remove_stopwords(text):
    tokens = re.findall(r"\b[a-z]{3,}\b", text)
    return ' '.join(t for t in tokens if t not in STOP)

df['hashtags']       = df['review_text'].apply(extract_hashtags)
df['cleaned_review'] = df['review_text'].apply(clean_text)
df['tokens']         = df['cleaned_review'].apply(remove_stopwords)

example = df[df['platform'] == 'Twitter'].iloc[2]
print('RAW    :', example['review_text'])
print('CLEANED:', example['cleaned_review'])
print('TOKENS :', example['tokens'])
print('HASHTAG:', example['hashtags'])


# ==========================================================================================
# ### 1.4 Handle missing data
# The brief suggests `df.dropna()`. On the combined table that would delete every row, because Instagram legitimately has no `date`, `username` or `retweets`. So I drop rows only when a **critical** field is missing (the text, or the age group needed for the age analysis), and remove duplicate review IDs and rows whose text is empty after cleaning.
# ==========================================================================================
before = len(df)
df = df.dropna(subset=['review_text', 'age_group'])          # critical fields only
df = df.drop_duplicates(subset='review_id')                  # same review twice
df = df[df['cleaned_review'].str.len() > 0].reset_index(drop=True)   # nothing left after cleaning
print(f'Rows before: {before:,} | after: {len(df):,} | dropped: {before - len(df):,}')

print('\nRemaining missing values (expected: only platform-specific fields):')
print(df.isna().sum()[lambda s: s > 0].to_string())


# ==========================================================================================
# ## Part 2 — Sentiment Analysis
# ### 2.1 VADER compound scores and classification
# VADER's **compound** score runs from -1 (most negative) to +1 (most positive). As in the brief: `> 0.05` is **Positive**, `< -0.05` is **Negative**, and anything in between is **Neutral**.
# ==========================================================================================
sid = SentimentIntensityAnalyzer()

def compound(text):
    return sid.polarity_scores(text)['compound']

def classify_sentiment(score):
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    return 'Neutral'

df['sentiment_score'] = df['cleaned_review'].apply(compound)
df['sentiment']       = df['sentiment_score'].apply(classify_sentiment)

print(df.groupby('platform')['sentiment'].value_counts(normalize=True).mul(100).round(1)
        .unstack().reindex(columns=SENT_ORDER).to_string())


# ==========================================================================================
# ### 2.2 Robustness check: does cleaning change the result?
# VADER is designed for raw social-media text (it reads capitals, `!!!` and emoji). I re-score the **original, uncleaned** text and compare the labels.
# **Result:** the two methods agree on 98.6% of rows. All 56 disagreements are the same tweet ("...Eye-opening and terrifying....", sometimes `terrifying.!!!`). On the raw text VADER sees the token `terrifying....` with the dots glued to it, does not recognise the word, and scores it Neutral. After cleaning, the word is recognised and the tweet is correctly Negative. So cleaning here fixes a real scoring error.
# ==========================================================================================
df['sentiment_score_raw'] = df['review_text'].apply(compound)
df['sentiment_raw']       = df['sentiment_score_raw'].apply(classify_sentiment)

agree = (df['sentiment'] == df['sentiment_raw'])
print(f'Label agreement, cleaned vs raw text: {agree.mean():.1%}')
print(df.loc[~agree].groupby(['platform', 'sentiment', 'sentiment_raw']).size()
        .rename('rows').reset_index().to_string(index=False))


# ==========================================================================================
# ### 2.3 What does VADER actually say about each message?
# Both datasets are built from a small set of repeated messages, so it is worth reading VADER's verdict on every distinct one. This is the most honest way to judge the model. Any message that looks mislabelled to a human reader is a limitation to report (see Part 4).
# ==========================================================================================
msg = (df.groupby(['platform', 'cleaned_review'])
         .agg(n=('review_id', 'size'), score=('sentiment_score', 'first'), label=('sentiment', 'first'))
         .reset_index().sort_values(['platform', 'score'], ascending=[True, False]))
pd.set_option('display.max_colwidth', 90, 'display.width', 200)
display(msg.rename(columns={'cleaned_review': 'message (cleaned)'}).reset_index(drop=True))


# ==========================================================================================
# ### 2.4 Topic tagging
# To answer *"what are the most common topics?"* I tag every row with a topic using simple keyword rules. The datasets are template-based, so rules are accurate and fully transparent here.
# ==========================================================================================
def tag_topic(row):
    t = row['cleaned_review']
    if row['platform'] == 'Instagram':
        if 'delivery' in t:                       return 'Delivery & quality'
        if re.search(r'\bapp\b', t):              return 'App experience'
        if 'service' in t:                        return 'Customer service'
        if 'product' in t or 'purchasing' in t:   return 'Product & value'
        return 'General experience'
    # Twitter (climate / environment)
    if 'renewable' in t:                          return 'Renewable energy'
    if 'electric vehicles' in t:                  return 'Electric vehicles'
    if 'sea levels' in t:                         return 'Rising sea levels'
    if 'plant trees' in t:                        return 'Tree planting / volunteering'
    if 'forests' in t:                            return 'Deforestation'
    if 'plastic' in t:                            return 'Plastic waste'
    if 'recycle' in t:                            return 'Recycling doubts'
    if 'litter' in t:                             return 'Littering'
    if 'deny climate' in t:                       return 'Climate denial'
    if 'climate change is real' in t:             return 'Climate awareness'
    return 'Other'

df['topic'] = df.apply(tag_topic, axis=1)
print(df.groupby('platform')['topic'].value_counts().to_string())
print('\nUntagged rows:', (df['topic'] == 'Other').sum())


# ==========================================================================================
# ### 2.5 Save the processed datasets
# ==========================================================================================
out_cols = ['review_id', 'user_id', 'username', 'platform', 'age_group', 'location', 'date',
            'review_text', 'cleaned_review', 'hashtags', 'topic',
            'likes', 'retweets', 'replies',
            'sentiment_score', 'sentiment', 'sentiment_score_raw', 'sentiment_raw']

df[out_cols].to_csv('processed_social_media_reviews.csv', index=False)          # combined (main deliverable)
(df[df['platform'] == 'Instagram'][out_cols].dropna(axis=1, how='all')
   .to_csv('processed_instagram_reviews.csv', index=False))
(df[df['platform'] == 'Twitter'][out_cols].dropna(axis=1, how='all')
   .to_csv('processed_twitter_tweets.csv', index=False))
print('Saved: processed_social_media_reviews.csv, processed_instagram_reviews.csv, processed_twitter_tweets.csv')
display(df[out_cols].head())


# ==========================================================================================
# ## Part 3 — Data Visualisation
# ### 3.1 Overall sentiment distribution
# ==========================================================================================
ig_df = df[df['platform'] == 'Instagram']
tw_df = df[df['platform'] == 'Twitter'].copy()

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, (title, sub) in zip(axes, [('Combined', df), ('Instagram', ig_df), ('Twitter', tw_df)]):
    counts = sub['sentiment'].value_counts().reindex(SENT_ORDER).fillna(0)
    bars = ax.bar(counts.index, counts.values, color=[SENT_COLORS[s] for s in counts.index], width=0.65)
    ax.bar_label(bars, labels=[f'{int(v):,}\n({v / len(sub):.1%})' for v in counts.values], padding=3, fontsize=10)
    ax.set_title(f'{title} (n = {len(sub):,})', fontweight='bold')
    ax.set_xlabel(''); ax.set_ylabel('Number of posts')
    ax.set_ylim(0, counts.max() * 1.22)
fig.suptitle('Sentiment Distribution', fontsize=15, fontweight='bold', y=1.03)
save_fig(fig, '01_sentiment_distribution')


# ==========================================================================================
# ### 3.2 Sentiment by age group (counts and share of each group)
# ==========================================================================================
def plot_age_sentiment(sub, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))

    # Left: grouped bars of counts (as in the brief)
    sns.countplot(data=sub, x='age_group', hue='sentiment', order=AGE_ORDER,
                  hue_order=SENT_ORDER, palette=SENT_COLORS, ax=axes[0])
    for c in axes[0].containers:
        axes[0].bar_label(c, fontsize=8, padding=2)
    axes[0].set_title('Number of posts', fontweight='bold')
    axes[0].set_xlabel(''); axes[0].set_ylabel('Count')
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.15); axes[0].legend(title='Sentiment', loc='upper left', ncol=3, fontsize=8)

    # Right: 100% stacked bars (fair comparison when group sizes differ)
    pct = (pd.crosstab(sub['age_group'], sub['sentiment'], normalize='index') * 100
             ).reindex(index=AGE_ORDER, columns=SENT_ORDER)
    pct.plot(kind='bar', stacked=True, ax=axes[1], width=0.75,
             color=[SENT_COLORS[s] for s in SENT_ORDER])
    pct_labels(axes[1])
    axes[1].set_title('Share within each age group', fontweight='bold')
    axes[1].set_xlabel(''); axes[1].set_ylabel('% of posts'); axes[1].legend(title='Sentiment', loc='upper left', bbox_to_anchor=(1.01, 1))
    axes[1].tick_params(axis='x', rotation=0)

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    save_fig(fig, filename)
    return pct

pct_ig  = plot_age_sentiment(ig_df, 'Sentiment by Age Group — Instagram', '02a_age_group_instagram')
pct_tw  = plot_age_sentiment(tw_df, 'Sentiment by Age Group — Twitter',   '02b_age_group_twitter')
pct_all = plot_age_sentiment(df,    'Sentiment by Age Group — Combined',  '02c_age_group_combined')


# ==========================================================================================
# ### 3.3 Word clouds
# ==========================================================================================
PALETTES = {'Greens': ['#14532d', '#166534', '#15803d', '#1b8a47'],
            'Reds':   ['#7f1d1d', '#991b1b', '#b91c1c', '#c0392b']}

def make_wc(text, colormap):
    """Word cloud with dark, high-contrast colours so even the biggest words stay readable."""
    if not text.strip():
        return None
    palette = PALETTES[colormap]
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return palette[int(random_state.random() * len(palette))]
    return WordCloud(width=1000, height=520, background_color='white', color_func=color_func,
                     max_words=80, collocations=False, random_state=42).generate(text)

# (a) Positive reviews — combined (the brief's required chart)
pos_text = ' '.join(df.loc[df['sentiment'] == 'Positive', 'tokens'])
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.imshow(make_wc(pos_text, 'Greens'), interpolation='bilinear'); ax.axis('off')
ax.set_title('Most Frequent Words in Positive Posts (Instagram + Twitter)', fontsize=14, fontweight='bold')
save_fig(fig, '03a_wordcloud_positive')

# (b) Positive vs negative, by platform
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
for r, senti, cmap in [(0, 'Positive', 'Greens'), (1, 'Negative', 'Reds')]:
    for c, plat in enumerate(['Instagram', 'Twitter']):
        text = ' '.join(df.loc[(df['sentiment'] == senti) & (df['platform'] == plat), 'tokens'])
        wc = make_wc(text, cmap)
        if wc is not None:
            axes[r, c].imshow(wc, interpolation='bilinear')
        axes[r, c].axis('off')
        axes[r, c].set_title(f'{senti} — {plat}', fontweight='bold')
fig.suptitle('Word Clouds by Sentiment and Platform', fontsize=15, fontweight='bold')
plt.tight_layout()
save_fig(fig, '03b_wordclouds_by_platform')

# (c) Top words in positive posts, as a bar chart (easier to read exact counts than a cloud)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
top_words = {}
for ax, plat in zip(axes, ['Instagram', 'Twitter']):
    words = Counter(' '.join(df.loc[(df['sentiment'] == 'Positive') & (df['platform'] == plat), 'tokens']).split())
    top = pd.Series(dict(words.most_common(12)))[::-1]
    top_words[plat] = top[::-1]
    ax.barh(top.index, top.values, color=PLATFORM_COLORS[plat])
    ax.set_title(f'Top words in Positive posts — {plat}', fontweight='bold')
    ax.set_xlabel('Occurrences')
save_fig(fig, '03c_top_positive_words')


# ==========================================================================================
# ### 3.4 Sentiment trends over time
# Only the **Twitter** file has dates (all of 2024). I use **monthly** aggregation. The brief's daily grouping gives 366 days x 3 lines, which is far too noisy to read.
# ==========================================================================================
tw_df['month'] = tw_df['date'].dt.to_period('M').dt.to_timestamp()

monthly     = (tw_df.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
                    .reindex(columns=SENT_ORDER, fill_value=0))
monthly_pct = monthly.div(monthly.sum(axis=1), axis=0) * 100
daily_score = tw_df.groupby('date')['sentiment_score'].mean()
rolling_30  = daily_score.rolling('30D', min_periods=10).mean()

fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
for s in SENT_ORDER:
    axes[0].plot(monthly.index, monthly[s], marker='o', lw=2.2, color=SENT_COLORS[s], label=s)
    axes[1].plot(monthly_pct.index, monthly_pct[s], marker='o', lw=2.2, color=SENT_COLORS[s], label=s)
axes[0].set_title('Number of tweets per month, by sentiment', fontweight='bold'); axes[0].set_ylabel('Tweets'); axes[0].legend()
axes[1].set_title('Share of tweets per month, by sentiment', fontweight='bold'); axes[1].set_ylabel('% of month'); axes[1].legend()
axes[2].scatter(daily_score.index, daily_score.values, s=10, color='#bbbbbb', label='Daily average')
axes[2].plot(rolling_30.index, rolling_30.values, lw=2.8, color='#1DA1F2', label='30-day rolling average')
axes[2].axhline(0, color='black', lw=0.8, ls='--')
axes[2].set_title('Average VADER compound score over time', fontweight='bold'); axes[2].set_ylabel('Compound score'); axes[2].legend()
import matplotlib.dates as mdates
axes[2].xaxis.set_major_locator(mdates.MonthLocator())
axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
fig.suptitle('Twitter Sentiment Trends, 2024', fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
save_fig(fig, '04_sentiment_trends_twitter')

display(monthly.assign(total=monthly.sum(axis=1), pct_positive=monthly_pct['Positive'].round(1),
                       pct_negative=monthly_pct['Negative'].round(1)))


# ==========================================================================================
# ### 3.5 Additional charts
# ==========================================================================================
# (a) Twitter: sentiment by country
ct = (pd.crosstab(tw_df['location'], tw_df['sentiment'], normalize='index') * 100).reindex(columns=SENT_ORDER)
ct = ct.sort_values('Positive')
ax = ct.plot(kind='barh', stacked=True, figsize=(11, 5.5), color=[SENT_COLORS[s] for s in SENT_ORDER], width=0.75)
pct_labels(ax)
ax.set_title('Twitter: Sentiment by Country (sorted by % positive)', fontweight='bold')
ax.set_xlabel('% of tweets'); ax.set_ylabel(''); ax.legend(title='Sentiment', loc='upper left', bbox_to_anchor=(1.01, 1))
save_fig(ax.get_figure(), '05_sentiment_by_country_twitter')

# (b) Engagement by sentiment
eng_long = df.melt(id_vars=['platform', 'sentiment'], value_vars=['likes', 'replies', 'retweets'],
                   var_name='metric', value_name='count').dropna(subset=['count'])
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, metric in zip(axes, ['likes', 'replies', 'retweets']):
    sns.barplot(data=eng_long[eng_long['metric'] == metric], x='sentiment', y='count', hue='platform',
                order=SENT_ORDER, palette=PLATFORM_COLORS, errorbar='se', ax=ax)
    ax.set_title(f'Average {metric}', fontweight='bold'); ax.set_xlabel(''); ax.set_ylabel('Mean (± SE)')
    if ax.get_legend(): ax.legend(title='')
fig.suptitle('Engagement by Sentiment (retweets exist only on Twitter)', fontsize=14, fontweight='bold', y=1.03)
save_fig(fig, '06_engagement_by_sentiment')
engagement_table = df.groupby(['platform', 'sentiment'])[['likes', 'replies', 'retweets']].mean().round(1)
display(engagement_table)

# (c) Sentiment by topic, per platform
fig, axes = plt.subplots(1, 2, figsize=(17, 6))
topic_pct = {}
for ax, plat in zip(axes, ['Instagram', 'Twitter']):
    sub = df[df['platform'] == plat]
    t = (pd.crosstab(sub['topic'], sub['sentiment'], normalize='index') * 100).reindex(columns=SENT_ORDER, fill_value=0)
    t = t.sort_values('Positive')
    topic_pct[plat] = t
    t.plot(kind='barh', stacked=True, ax=ax, color=[SENT_COLORS[s] for s in SENT_ORDER], width=0.75)
    pct_labels(ax, 8)
    ax.set_title(f'{plat}: sentiment by topic', fontweight='bold'); ax.set_xlabel('% of posts'); ax.set_ylabel('')
    ax.legend(title='Sentiment', loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
plt.tight_layout()
save_fig(fig, '07_sentiment_by_topic')

# Topics that dominate the POSITIVE posts
pos_topics = (df[df['sentiment'] == 'Positive'].groupby(['platform', 'topic']).size()
                .rename('positive_posts').reset_index()
                .sort_values(['platform', 'positive_posts'], ascending=[True, False]))
display(pos_topics)

# (d) VADER score for every distinct message (transparency chart)
m = (df.groupby('cleaned_review')
       .agg(n=('review_id', 'size'), score=('sentiment_score', 'first'),
            label=('sentiment', 'first'), platform=('platform', 'first'))
       .reset_index().sort_values('score'))
m['wrapped'] = m['cleaned_review'].str.capitalize().apply(lambda s: textwrap.fill(s, 58))

fig, ax = plt.subplots(figsize=(12, 13))
ax.barh(m['wrapped'], m['score'], color=[SENT_COLORS[l] for l in m['label']])
for y, (sc, n, plat) in enumerate(zip(m['score'], m['n'], m['platform'])):
    ax.text(sc + (0.02 if sc >= 0 else -0.02), y, f'{sc:+.2f}  (n={n}, {plat[:2]})',
            va='center', ha='left' if sc >= 0 else 'right', fontsize=8)
ax.axvline(0.05, color='black', ls=':', lw=0.8); ax.axvline(-0.05, color='black', ls=':', lw=0.8)
ax.set_xlim(-1.15, 1.15); ax.tick_params(axis='y', labelsize=8)
ax.set_title('VADER compound score for each distinct message\n(dotted lines = ±0.05 classification thresholds)', fontweight='bold')
ax.set_xlabel('Compound score')
save_fig(fig, '08_score_by_message')


# ==========================================================================================
# ## Part 4 — Statistical checks and insights
# ### 4.1 Are the differences real? (chi-square tests of independence)
# Bar charts always show *some* difference between groups. A chi-square test checks whether those differences are bigger than random variation would produce. **Cramér's V** is the effect size (0 = no association, 1 = perfect association).
# ==========================================================================================
def chi2_row(sub, col, label):
    ct = pd.crosstab(sub[col], sub['sentiment'])
    chi2, p, dof, _ = chi2_contingency(ct)
    n = ct.values.sum()
    v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    return {'test': label, 'chi2': round(chi2, 2), 'dof': dof, 'p_value': round(p, 4),
            "Cramér's V": round(v, 3), 'significant (p<0.05)': 'YES' if p < 0.05 else 'no'}

tests = pd.DataFrame([
    chi2_row(ig_df, 'age_group', 'Instagram: age group x sentiment'),
    chi2_row(tw_df, 'age_group', 'Twitter: age group x sentiment'),
    chi2_row(df,    'age_group', 'Combined: age group x sentiment'),
    chi2_row(tw_df, 'location',  'Twitter: country x sentiment'),
    chi2_row(tw_df, 'month',     'Twitter: month x sentiment'),
    chi2_row(df,    'platform',  'Platform x sentiment'),
])
display(tests)


# ==========================================================================================
# ### 4.2 Key numbers behind the insights
# ==========================================================================================
def share(sub):
    return sub['sentiment'].value_counts(normalize=True).mul(100).reindex(SENT_ORDER).fillna(0).round(1)

print('OVERALL SENTIMENT SHARE (%)')
print(pd.DataFrame({'Combined': share(df), 'Instagram': share(ig_df), 'Twitter': share(tw_df)}).to_string())

for name, pct in [('INSTAGRAM', pct_ig), ('TWITTER', pct_tw), ('COMBINED', pct_all)]:
    print(f'\n{name} — % within age group')
    print(pct.round(1).to_string())
    print(f'  Most positive group: {pct["Positive"].idxmax()} ({pct["Positive"].max():.1f}%) | '
          f'Most negative group: {pct["Negative"].idxmax()} ({pct["Negative"].max():.1f}%)')

print('\nTWITTER MONTHLY % POSITIVE / NEGATIVE')
print(monthly_pct[['Positive', 'Negative']].round(1).T.to_string())
print(f'  Highest positive month: {monthly_pct["Positive"].idxmax():%B} ({monthly_pct["Positive"].max():.1f}%) | '
      f'Lowest: {monthly_pct["Positive"].idxmin():%B} ({monthly_pct["Positive"].min():.1f}%)')

print('\nTOP WORDS IN POSITIVE POSTS')
for plat, s in top_words.items():
    print(f'  {plat}:', ', '.join(f'{w} ({c})' for w, c in s.head(8).items()))

print('\nTOP HASHTAGS (Twitter) and % positive')
tags = tw_df.assign(tag=tw_df['hashtags'].str.split()).explode('tag').dropna(subset=['tag'])
print(tags.groupby('tag')['sentiment'].agg(posts='size', pct_positive=lambda s: round((s == 'Positive').mean() * 100, 1))
          .sort_values('posts', ascending=False).to_string())


# ==========================================================================================
# ### 4.3 Findings
# The written findings are in `Insights_Report.md`, with the numbers tied to the tables above. In short:
# - **Overall sentiment** and **age-group** results are reported per platform and combined, with the chi-square tests above telling you whether any age-group difference is statistically meaningful.
# - **Time trends** are Twitter-only (the Instagram file has no dates).
# - **Topics:** the positive-post tables and word clouds show which themes drive positive sentiment.
# - **Limitations:** VADER is a general-purpose lexicon model. A few messages in these datasets are labelled differently from what a human reader would choose (see section 2.3 and chart `08_score_by_message`). In particular, mildly lukewarm messages such as *"It was okay. Nothing special but not bad either."* score as Positive because VADER reads "okay" and "not bad" as positive cues. Messages built around a negative-looking word such as *"waste"* in *"We need to do more to reduce plastic waste"* score as Negative even though the intent is a call to action.
# ==========================================================================================