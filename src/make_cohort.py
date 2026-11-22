# Import the os module, which lets Python create and check folders
import os
# Import numpy for fast random sampling and array math
import numpy as np
# Import pandas to read the frequency table and save the cohort as a table
import pandas as pd
# Import matplotlib's plotting module to draw a chart
import matplotlib.pyplot as plt

# Which published source to use:"riaz2019" vs "pk2021" (alleles *1, *2, *3); ran 1 first then 2nd to compare
SCENARIO = "pk2021"
# Number of replicate cohorts; each one gets its own draw of allele frequencies
N_REPLICATES = 10
# Number of virtual people per replicate (10 x 100,000 = 1,000,000 people)
PEOPLE_PER_REPLICATE = 100_000
# Smaller = more uncertainty; 200 reflects that Pakistani studies disagree with each other more than sampling error alone suggests (an assumption for the cohort card)
EFFECTIVE_N_CHROMOSOMES = 200
# Seed for the random number generator so re-running gives the same cohort
SEED = 42

# Create the data folder if it doesn't already exist (exist_ok=True means no error if it's already there)
os.makedirs("data", exist_ok=True)
# Create the outputs folder if it doesn't already exist
os.makedirs("outputs", exist_ok=True)

# Read the allele frequency table from the data folder into a table called "freq"
freq = pd.read_csv("data/allele_frequencies.csv")
# Keep only CYP2C19 rows from the chosen scenario (the two studies are never blended because they used different cohorts)
freq = freq[(freq["gene"] == "CYP2C19") & (freq["scenario"] == SCENARIO)].reset_index(drop=True)
# Put the allele names (*1, *2, ...) into a plain array
alleles = freq["allele"].to_numpy()
# Put the published frequencies into an array of decimal numbers
base_freq = freq["frequency"].to_numpy(dtype=float)
# Put each allele's function label (normal, no_function, increased) into an array
functions = freq["function"].to_numpy()

# Safety check: the frequencies must add up to about 1, otherwise stop with a message
assert np.isclose(base_freq.sum(), 1.0, atol=0.01), "Allele frequencies must sum to 1"
# Rescale so they add up to exactly 1 (removes tiny rounding errors)
base_freq = base_freq / base_freq.sum()

# Create the random number generator from our seed
rng = np.random.default_rng(SEED)
# Make an empty list that will collect one table per replicate
pieces = []

# Loop over replicate numbers 0, 1, 2, ... 9
for rep in range(N_REPLICATES):
    # Draw one plausible set of population allele frequencies around the published ones (a Dirichlet draw is a random split of 100% across the alleles)
    rep_freq = rng.dirichlet(base_freq * EFFECTIVE_N_CHROMOSOMES)
    # Pick an allele for each person's first chromosome using this replicate's frequencies
    a1 = rng.choice(len(alleles), size=PEOPLE_PER_REPLICATE, p=rep_freq)
    # Pick an allele for each person's second chromosome the same way (assumes Hardy-Weinberg: the two copies are independent)
    a2 = rng.choice(len(alleles), size=PEOPLE_PER_REPLICATE, p=rep_freq)
    # Look up the function label of each person's first allele
    f1 = functions[a1]
    # Look up the function label of each person's second allele
    f2 = functions[a2]
    # Count each person's no-function alleles (0, 1 or 2)
    n_no = (f1 == "no_function").astype(int) + (f2 == "no_function").astype(int)
    # Count each person's increased-function alleles (0, 1 or 2)
    n_inc = (f1 == "increased").astype(int) + (f2 == "increased").astype(int)
    # Start everyone as NM (normal metabolizer); the next lines overwrite this for other groups
    phenotype = np.full(PEOPLE_PER_REPLICATE, "NM", dtype=object)
    # One increased allele and no no-function alleles = RM (rapid metabolizer)
    phenotype[(n_no == 0) & (n_inc == 1)] = "RM"
    # Two increased alleles and no no-function alleles = UM (ultrarapid metabolizer)
    phenotype[(n_no == 0) & (n_inc == 2)] = "UM"
    # Exactly one no-function allele = IM (intermediate metabolizer)
    phenotype[n_no == 1] = "IM"
    # Two no-function alleles = PM (poor metabolizer)
    phenotype[n_no == 2] = "PM"
    # Take the lower allele index of each pair so diplotypes print consistently (*1/*2, never *2/*1)
    lo = np.minimum(a1, a2)
    # Take the higher allele index of each pair
    hi = np.maximum(a1, a2)
    # Build this replicate's table and add it to the list
    pieces.append(pd.DataFrame({
        # Which replicate each person belongs to
        "replicate": rep,
        # The person's first allele name
        "allele_1": alleles[lo],
        # The person's second allele name
        "allele_2": alleles[hi],
        # The genotype written as one string, like *1/*2
        "diplotype": alleles[lo] + "/" + alleles[hi],
        # The metabolizer phenotype worked out above
        "phenotype": phenotype,
    }))

# Stack the 10 replicate tables into one big table of 1,000,000 people
cohort = pd.concat(pieces, ignore_index=True)
# Add a unique ID number (0 to 999,999) as the first column
cohort.insert(0, "person_id", np.arange(len(cohort)))
# Save the cohort to a CSV file whose name includes the scenario, so the two scenarios don't overwrite each other
cohort.to_csv(f"data/cohort_cyp2c19_{SCENARIO}.csv", index=False)

# The order we want phenotypes displayed in
order = ["PM", "IM", "NM", "RM", "UM"]
# Count people per phenotype inside each replicate (replicates as rows, phenotypes as columns); phenotypes with nobody show as 0
counts = cohort.groupby(["replicate", "phenotype"]).size().unstack(fill_value=0).reindex(columns=order, fill_value=0)
# Convert the counts to percentages of each replicate
pct = counts.div(counts.sum(axis=1), axis=0) * 100
# Summarize across replicates: the mean, minimum and maximum percentage for each phenotype
summary = pd.DataFrame({"mean_%": pct.mean(), "min_%": pct.min(), "max_%": pct.max()}).round(2)
# Print which scenario this run used
print("Scenario:", SCENARIO)
# Print the summary so you can read it in the terminal
print(summary)

# Create an empty chart (fig is the page, ax is the plot area)
fig, ax = plt.subplots(figsize=(7, 4))
# Get the mean percentages as an array
means = summary["mean_%"].to_numpy()
# Work out how far each bar's lower error mark sits below the mean
err_low = means - summary["min_%"].to_numpy()
# Work out how far each bar's upper error mark sits above the mean
err_high = summary["max_%"].to_numpy() - means
# Draw one bar per phenotype with error marks showing the spread across replicates
ax.bar(order, means, yerr=[err_low, err_high], capsize=4)
# Label the vertical axis
ax.set_ylabel("% of synthetic cohort")
# Give the chart a two-line title that names the scenario and explains the error bars
ax.set_title(f"CYP2C19 phenotypes, source: {SCENARIO}\n(bars: min-max of {N_REPLICATES} replicates)")
# Tidy the spacing so labels don't get cut off
fig.tight_layout()
# Save the chart as an image whose name includes the scenario
fig.savefig(f"outputs/cyp2c19_phenotypes_{SCENARIO}.png", dpi=150)
# Tell you the script finished
print(f"Done: saved data/cohort_cyp2c19_{SCENARIO}.csv and outputs/cyp2c19_phenotypes_{SCENARIO}.png")