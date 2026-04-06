# Kid-Level Code Walkthrough

This repository is like a big feelings science project.

It takes a huge pile of Reddit writing from people talking about grad school applications, asks a robot to guess how upset the writing sounds, finds the really bad "ouch" moments, and then checks whether those people seem more upset in the next couple of weeks than similar people who did not have that bad moment.

The code is mostly in Jupyter notebooks, so instead of one giant program, it is a row of recipe books. Each notebook does one job, saves a new file, and hands that file to the next notebook.

## The Whole Story

Imagine:

- `data/raw` is a giant toy box full of Reddit posts and comments.
- `data/processed` is the same toy box, but now everything is sorted into neat bags.
- `models` stores trained "sorting robots."
- `figures` stores the pictures the project draws at the end.
- `docs` explains the plan in words.

The pipeline is:

1. Read all the Reddit text.
2. Give each post a simple "sounds negative or not?" score.
3. Find the especially sad/scared posts.
4. Train smarter text robots that detect mental-health-like writing.
5. Count how many different subreddit "rooms" each user visits.
6. Compare before-vs-after changes fairly.

The main repo overview is in `README.md`, and the intended flow is documented in `docs/flow.md` and `docs/pipeline.md`.

## Folder By Folder

### `data/raw`

This is where the original Reddit files go.

- They are not in the repo because they are big.
- One file is posts, one file is comments.

### `data/processed`

These are the saved results from each notebook.

- Think of them as checkpoints, so the project does not need to redo everything every time.

### `models`

This holds the trained classifiers.

- A classifier is just a robot that learned to say, "this writing looks more like stress/depression/anxiety writing" or "this looks less like that."

### `figures`

This holds the final charts and plots.

- These pictures help show the answer visually.

### `docs`

These explain what the notebooks do, what the research question is, and what the final results mean.

## Notebook 00: Small Practice Sandbox

File: `notebooks/00_exploratory_topic_sentiment.ipynb`

This notebook is not the main pipeline. It is more like a warm-up playground.

What it does:

- Loads one dataset, especially September 2024 data.
- Combines title + body text.
- Filters to just that month.
- Does topic modeling.
- Does simple sentiment analysis.

In kid words:

- It asks, "What are people talking about this month?"
- Then it asks, "Which topics sound happiest or saddest?"

Important parts:

- `CountVectorizer`: turns words into countable pieces.
- `LatentDirichletAllocation`: groups posts into hidden topics.
- `VADER`: gives each text a sentiment score.
- Then it plots bars and examples.

This notebook is exploratory. It helps the researcher peek around, but it is not the main machine that answers the paper's big question.

## Notebook 01: Score The Whole Corpus With VADER

File: `notebooks/01_score_corpus.ipynb`

This is the first real production step.

What it loads:

- All posts
- All comments

What it does:

1. Reads JSONL files into pandas tables.
2. Combines posts and comments into one big table.
3. Parses dates.
4. Creates a `week` column, because later the project compares week-by-week.
5. Removes junk users like deleted accounts and bots.
6. Makes sure text is not empty.
7. Runs VADER sentiment scoring on every row.

What VADER gives:

- `vader_compound`: one overall feeling number
- `vader_neg`: how negative the text sounds
- `vader_neu`: how neutral it sounds
- `vader_pos`: how positive it sounds

Then it makes helper columns:

- `distress_score = vader_neg`
- a text label like positive / neutral / negative
- `year_month`

Then it saves the result to:

- `data/processed/scored_corpus.parquet`

Why this exists:

- It gives every single Reddit row a first-pass feelings score.
- It is simple and fast.

Kid version:

- This notebook teaches a small robot to put a "happy/sad sticker" on every note in the toy box.

Important limitation:

- VADER is dumb-simple.
- If a post says "rejected," VADER often treats that as negative even if the post is not really emotional.
- So later the repo builds a smarter robot.

## Notebook 02: Find The Important Sad Events

File: `notebooks/02_anchor_events.ipynb`

This notebook takes the VADER-scored data and says:
"Which posts count as the big bad event we care about?"

Those are called **anchor posts**.

What it does:

1. Loads the scored corpus.
2. Looks only at posts, not comments, for anchor finding.
3. Builds a list of regex patterns like:
   - rejection
   - declined
   - waitlisted
   - funding lost
   - anxiety
   - depressed
   - stress
   - falling apart
   - can't cope

Then it marks a post as an anchor if:

- it has one of those keywords, and
- it also looks distressed enough by VADER thresholds

It also lets some self-labeled rejected/waitlisted posts count if they pass the distress rule.

Then it saves:

- `anchor_posts.parquet`

Next, it builds weekly user summaries:

- for each `author` and `week`
- count how many things they posted
- average their sentiment/distress values

That becomes:

- `user_weekly_scores.parquet`

Then it creates exposure labels:

- `exposed`: a user who wrote an anchor post that week
- `unexposed`: a user active that same week who did not write one

That becomes:

- `exposure_labels.parquet`

Kid version:

- This notebook looks through all the notes and circles the ones that feel like "something really painful happened here."
- Then it makes two teams:
- Team A: kids who had that bad moment
- Team B: kids who were around that week but did not

Why this matters:

- The whole research question depends on identifying the "event."
- No anchor posts, no comparison.

## Notebook 03: Train Smarter Feelings Robots

File: `notebooks/03_train_classifiers.ipynb`

This is one of the most important notebooks.

Notebook 01 used VADER, which is a generic rulebook.
Notebook 03 builds custom classifiers that are better at recognizing writing that looks like anxiety, depression, and stress.

What it does first:

- Pulls training data from Arctic Shift API.
- Positive examples come from mental-health subreddits:
  - `anxiety`
  - `depression`
  - `stress`
- Negative examples come from more ordinary control subreddits:
  - `personalfinance`
  - `learnprogramming`
  - `todayilearned`
  - `careerguidance`

Why:

- The notebook wants the robot to learn the *style* of distressed writing, not just negative words.

Then it trains three separate models:

- anxiety classifier
- depression classifier
- stress classifier

Important tools:

- `TfidfVectorizer`: turns text into weighted word features
- `LinearSVC`: the actual classifier

Kid version:

- The notebook shows the robot lots of examples of "this sounds like anxious writing" and "this sounds like normal non-mental-health writing."
- After enough examples, the robot gets better at guessing.

Then it evaluates the models on held-out test samples.

After training, it scores the full gradadmissions corpus:

- each row gets:
  - `anx_score`
  - `dep_score`
  - `str_score`

Then it makes:

- `mh_score = average(anx_score, dep_score, str_score)`

This is the main distress measure used later.

Then it saves:

- model files in `models/clf_*.joblib`
- `data/processed/scored_corpus_v2.parquet`
- `data/processed/user_weekly_scores_v2.parquet`

Why this notebook exists:

- It replaces the simple "sad word counter" with a smarter "does this writing look like distressed writing?" detector.

That is a big deal in this repo.

## Notebook 04: Count How Wide A User's Reddit World Is

File: `notebooks/04_collect_community_breadth.ipynb`

This notebook measures **community breadth**.

That means:
"How many different subreddits does this person participate in, besides r/GradAdmissions?"

What it does:

1. Loads the panel users.
2. Loads a checkpoint file if one exists.
3. For each user, calls the Arctic Shift API.
4. Gets the list of subreddits they interacted in.
5. Counts distinct subreddits.
6. Saves progress every so often so it can resume later.

Output:

- `user_community_breadth.parquet`
- plus a checkpoint JSONL file

Kid version:

- It checks whether a user only hangs out in one room, or runs around to many rooms.

Why this matters:

- One research question asks whether people with a wider Reddit life are more protected from stress.
- The hypothesis was: maybe more communities means more support.
- The result ended up not supporting that.

Important code idea:

- It uses `requests.Session()` and rate-limit handling so it does not hit the API too fast.

## Notebook 05: The Main Comparison

File: `notebooks/05_did_analysis_v2.ipynb`

This is the main analysis notebook. If the repo has a "final boss," this is it.

It uses:

- the smarter `mh_score`
- the exposure labels
- the community breadth data

Its job is:
"After the anchor event, do exposed users change more than similar unexposed users?"

That sounds fancy, but the idea is simple.

First, it builds a small time window around each event week:

- 2 weeks before
- 2 weeks after
- usually skipping the exact event week itself

So each user-event pair becomes a tiny timeline.

Then it does **propensity score matching**.

Kid version of matching:

- If one kid is already super upset before the event, and another kid is usually very calm, that is not a fair comparison.
- So the notebook tries to pair kids who looked similar *before* the event.

How it does that:

- Uses features like pre-event `mh_score` and posting activity
- Fits logistic regression to estimate a propensity score
- Uses nearest-neighbor matching with a caliper

In plain words:

- It gives each user a "how likely were they to be exposed?" number
- Then it matches exposed users to unexposed users with similar numbers

After matching, it runs the main regression.

The key term is:

- `post × exposed`

Kid meaning:

- "Did Team A change more after the event than Team B did?"

That is the core causal estimate.

Then the notebook does RQ2:

- adds community breadth
- especially the three-way interaction with `post × exposed × breadth`

Kid meaning:

- "Does having more subreddit rooms change how big the after-effect is?"

Then it draws event-study plots:

- before the event, the two groups should look similar
- after the event, they separate

Why that matters:

- It is a fairness check.
- If they already looked wildly different before the event, the conclusion would be weaker.

Then it compares:

- VADER-based analysis
- SVM-based analysis

Then it runs the analysis separately by admissions cycle:

- 2023-24
- 2024-25

Kid version:

- It asks, "Did the result happen in both school years, or was it just one weird year?"

This notebook produces the main answer of the paper.

## Notebook 06: Simpler Baseline Version

File: `notebooks/06_did_analysis_vader_baseline.ipynb`

This notebook is like Notebook 05's simpler cousin.

It does many of the same steps:

- build a pre/post panel
- match exposed and unexposed users
- run DiD regression
- draw event-study plots
- test moderation by community breadth
- check seasonality
- check replication by cycle

But instead of using the smarter `mh_score`, it uses the simpler VADER distress measure.

Why keep this notebook?

- It gives a baseline comparison.
- It shows that the smarter classifier is better and more precise.

Kid version:

- This is the "use the old simpler robot and see how it does" notebook.

## The Big Ingredients In The Code

### `pandas`

The main table-handling tool.

- Almost every notebook uses it.
- It loads files, filters rows, groups by user/week, computes averages, and saves parquet files.

Kid version:

- It is the giant spreadsheet helper.

### `numpy`

Handles numbers and math bits.

### `matplotlib`

Draws charts.

### `requests`

Talks to the Arctic Shift API on the internet.

### `sklearn`

Machine learning tools:

- `CountVectorizer`
- `TfidfVectorizer`
- `LatentDirichletAllocation`
- `LinearSVC`
- `LogisticRegression`
- `NearestNeighbors`
- `StandardScaler`

Kid version:

- This is the toolbox full of robots and math helpers.

### `statsmodels`

Used for the regression analysis in the later notebooks.

Kid version:

- This is the "careful comparison calculator."

### `joblib`

Saves trained models to disk so they do not need retraining every time.

## The Important Saved Files

These are the handoff files that make the whole repo work:

`scored_corpus.parquet`

- First-pass VADER scores for every post/comment.

`anchor_posts.parquet`

- The posts selected as anchor events.

`exposure_labels.parquet`

- Who is exposed and who is unexposed in each event week.

`user_weekly_scores.parquet`

- Weekly averages using VADER-style measures.

`scored_corpus_v2.parquet`

- Full corpus plus smarter mental-health classifier scores.

`user_weekly_scores_v2.parquet`

- Weekly averages using the smarter `mh_score`.

`user_community_breadth.parquet`

- How many other subreddits each user participates in.

`clf_anxiety.joblib`, `clf_depression.joblib`, `clf_stress.joblib`

- The trained classifier robots.

## What The Stats Parts Mean In Baby Language

### "Classifier"

A robot sorter.

- It looks at words and guesses what kind of writing it is.

### "Feature"

A clue the robot uses.

- Example: certain words or word pairs.

### "Matching"

Pairing similar kids before comparing them.

### "Regression"

A careful math way to ask, "which pieces seem connected to the change?"

### "Difference-in-differences"

Not just "who is more upset?"

It is "who changed more after the important event?"

That matters because maybe everyone gets more stressed in admissions season anyway.

So instead of:

- "Are exposed users more distressed?"

the code asks:

- "Did exposed users go up more than similar unexposed users went up?"

That is much smarter.

## What The Repo Is Really Saying

In tiny-kid language:

- First, the repo reads lots of Reddit writing.
- Then it puts feeling scores on the writing.
- Then it finds the really painful posts.
- Then it teaches smarter robots to recognize distress-style writing.
- Then it checks whether people who had those painful posts seem worse in the next weeks than similar people who did not.
- Then it checks whether hanging out in more subreddits helps or not.

And the reported answer is:

- exposure to those distressed anchor posts is followed by a higher distress score
- and wider Reddit community breadth did not buffer the effect

## Simplest One-Sentence Summary

This code is a long assembly line that turns Reddit text into a fair before-vs-after comparison to test whether bad grad-admissions experiences are followed by more distressed writing.
