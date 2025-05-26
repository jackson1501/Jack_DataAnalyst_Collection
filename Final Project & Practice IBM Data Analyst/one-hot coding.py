import pandas as pd

# Load the data from the CSV file
df = pd.read_csv('group_employment.csv')

# --- Step 1: Prepare the DataFrame for exploding ---
# Create a copy of the original DataFrame to avoid modifying it directly.
# Reset the index and rename the new index column to 'original_index'
# This is crucial for correctly grouping the data back after exploding.
df_exploded = df.copy().reset_index().rename(columns={'index': 'original_index'})

# --- Step 2: Split the 'Employment' column into lists ---
# The 'Employment' column contains multiple values separated by semicolons.
# We split these strings into lists of individual employment types.
df_exploded['Employment'] = df_exploded['Employment'].str.split(';')

# --- Step 3: Explode the 'Employment' column ---
# The explode() method transforms each element of a list-like entry into a separate row.
# This means if an original row had ['Employed, full-time', 'Employed, part-time'],
# it will now become two rows, each with one employment type.
df_exploded = df_exploded.explode('Employment')

# --- Step 4: Perform one-hot encoding using get_dummies() ---
# pd.get_dummies() converts categorical data into dummy/indicator variables.
# For each unique employment type, a new column will be created.
# A '1' indicates the presence of that employment type in the row, '0' otherwise.
# The 'prefix' argument adds 'Employment_' to the new column names for clarity.
df_one_hot = pd.get_dummies(df_exploded['Employment'], prefix='Employment')

# --- Step 5: Concatenate and Aggregate ---
# We need to re-associate the one-hot encoded rows with their original entries.
# First, concatenate the 'original_index' column with the one-hot encoded DataFrame.
df_one_hot = pd.concat([df_exploded['original_index'], df_one_hot], axis=1)

# Then, group by the 'original_index' and sum the dummy variables.
# This effectively combines the exploded rows back into a single row for each original entry.
# If an original entry had multiple employment types, the corresponding dummy columns will sum to 1.
df_one_hot = df_one_hot.groupby('original_index').sum()

# --- Step 6: Display the results ---
# Print the first 5 rows of the final one-hot encoded DataFrame.
# .to_markdown() is used for clear, formatted output.
print("First 5 rows of one-hot encoded DataFrame:\n")
print(df_one_hot.head().to_markdown(numalign="left", stralign="left"))
