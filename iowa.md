# Iowa Gambling Task (IGT) Experiment

This is an experiment to automate the running, data collection, and analysis of the Iowa Gambling Task using the autoresearch framework.

## Setup

To set up a new IGT experiment:

1. Agree on a run tag: Propose a tag based on today's date (e.g., mar19). The branch igt/&lt;tag&gt; must not already exist.
2. Create the branch: `git checkout -b igt/&lt;tag&gt;` from the current main branch.
3. Read the in-scope files: The repo is small. Read these files for full context:
   - README.md — Repository context and IGT task description.
   - prepare.py — Fixed constants, task setup, and data logging. Do not modify.
   - run_igt.py — The file you modify. Task parameters, trial logic, and feedback mechanisms.
4. Verify data exists: Check that data/stimuli/ contains the IGT deck configurations. If not, tell the user to run `uv run prepare.py`.
5. Initialize results.tsv: Create results.tsv with just the header row. The baseline will be recorded after the first run.
6. Confirm and go: Confirm setup looks good.
7. Once confirmed, kick off the experimentation.

## Experimentation

Each IGT experiment runs for a fixed time budget of 10 minutes (wall clock time, excluding setup). Launch it as:

```bash
uv run run_igt.py
```

### What You CAN Do

Modify `run_igt.py` — This is the only file you edit. You can change:

- Task parameters (e.g., number of trials, deck configurations, reward/penalty schedules).
- Feedback mechanisms (e.g., visual/auditory feedback, delay durations).
- Data logging (e.g., reaction times, choices, subjective ratings).
- Analysis scripts (e.g., statistical tests, learning curves).

### What You CANNOT Do

- Modify prepare.py. It is read-only and contains fixed task logic, deck configurations, and evaluation metrics.
- Install new packages or add dependencies. Use only what's in pyproject.toml.
- Modify the core IGT logic (e.g., deck probabilities, win/loss structure).

### Goal

Optimize the task to maximize sensitivity to decision-making differences (e.g., between groups, conditions, or over time). The primary metric is net score (advantageous minus disadvantageous deck selections) over the last 20 trials.

### Constraints

- **Time**: Each run must complete within 10 minutes.
- **Complexity**: Prefer simpler, more interpretable changes. Avoid overly complex modifications unless they yield significant improvements.
- **Reproducibility**: Ensure the task logic remains consistent across runs.

## Output Format

After each run, the script prints a summary like this:

```text
---
net_score:         12.4
completion_time_s: 580.2
num_trials:        100
advantageous_choices: 65
disadvantageous_choices: 35
learning_rate:     0.72
```

## Logging Results

Log results to results.tsv (tab-separated) with 6 columns:

| commit | net_score | time_min | memory_gb | status | description |

Where:

- Git commit hash (short, 7 chars).
- Net score (e.g., 12.4).
- Completion time in minutes (round to 1 decimal place).
- Peak memory usage in GB (round to 1 decimal place).
- Status: keep, discard, or crash.
- Short description of the experiment (e.g., "increased penalty magnitude").

Example:

```text
commit    net_score    time_min    memory_gb    status    description
a1b2c3d    12.4    9.8    0.5    keep    baseline
b2c3d4e    15.1    10.0    0.6    keep    increased reward frequency
c3d4e5f    8.3    9.5    0.5    discard    removed feedback delay
```

## Experiment Loop

The experiment runs on a dedicated branch (e.g., igt/mar19).

**LOOP FOREVER:**

1. Check the current git state.
2. Modify run_igt.py with a new idea.
3. `git commit` your changes.
4. Run the experiment: `uv run run_igt.py > run.log 2>&1`.
5. Extract results: `grep "^net_score:\|^completion_time_s:\|^peak_vram_mb:" run.log`.
6. If the run crashes, check `tail -n 50 run.log` for errors. Fix and retry if trivial; otherwise, log as crash and move on.
7. Record results in results.tsv (do not commit this file).
8. If net_score improved, keep the commit. Otherwise, `git reset` to the previous state.

### Timeouts and Crashes

- **Timeout**: Kill runs exceeding 15 minutes. Treat as a failure.
- **Crashes**: Fix trivial issues (e.g., typos). For fundamental flaws, log as crash and discard.
- **Autonomy**: Never pause to ask for human input. If you run out of ideas, revisit the literature, combine past ideas, or explore radical changes.

## Example Workflow

1. **Baseline**: Run the default IGT configuration.
2. **Experiment**: Increase the penalty magnitude for disadvantageous decks.
3. **Analyze**: Compare net scores and learning curves.
4. **Iterate**: Test feedback delays, reward schedules, or visual cues.

## Adapting for Psychological Research

- **Integrate with Tools**: Use psychopy or jspsych for stimulus presentation.
- **Data Analysis**: Automate statistical tests (e.g., ANOVA, mixed-effects models) in run_igt.py.
- **Reproducibility**: Log all task parameters and random seeds.

## Key Files

### prepare.py (Read-Only)

- Defines deck configurations, win/loss probabilities, and fixed task logic.
- Contains the evaluation metric (net_score).

### run_igt.py (Editable)

- Implements the trial loop, feedback, and data logging.
- Place to modify task parameters and analysis.

## Why This Works for IGT

- **Automation**: Run hundreds of task variants unattended.
- **Reproducibility**: Track every change and result.
- **Optimization**: Focus on maximizing task sensitivity.
