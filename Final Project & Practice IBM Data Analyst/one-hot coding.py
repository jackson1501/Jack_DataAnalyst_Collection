import pandas as pd
import re
import os # Import the os module to check current working directory

# --- Helper to check current working directory ---
print(f"Current working directory: {os.getcwd()}")
print("Please ensure 'group_employment.csv' and 'unique_employment_phrases_final_refined_v2.csv' are in this directory, or update the paths below.")
print("-" * 80)

# Define the multi-word phrases and placeholder as used in the last refinement step
multi_word_phrases_to_preserve_space = [
    'and not looking for work',
    'independent contractor',
    'looking for work',
    'not employed',
    'i prefer not to say',
]
SPACE_PLACEHOLDER = '___TEMP_SPACE___'

# Function to normalize an employment string into a list of cleaned phrases
def normalize_employment_string(entry):
    entry_lower = str(entry).lower()

    # Temporarily replace spaces within the specified multi-word phrases with a placeholder
    for phrase in multi_word_phrases_to_preserve_space:
        entry_lower = entry_lower.replace(phrase, phrase.replace(' ', SPACE_PLACEHOLDER))

    # Normalize delimiters: replace commas with semicolons
    normalized_entry = entry_lower.replace(',', ';')

    # Split by semicolon to get major segments (values)
    segments = normalized_entry.split(';')

    cleaned_phrases_for_entry = []
    for segment in segments:
        # Split each segment by whitespace to get individual 'tokens'
        tokens = segment.split()

        for token in tokens:
            # Clean each token: allow letters, numbers, hyphens, and the temporary placeholder.
            cleaned_token = re.sub(f'[^{re.escape(SPACE_PLACEHOLDER)}a-z0-9-]', '', token).strip()

            if cleaned_token:
                # Restore the original spaces in the multi-word phrases
                final_phrase = cleaned_token.replace(SPACE_PLACEHOLDER, ' ')
                cleaned_phrases_for_entry.append(final_phrase)
    return cleaned_phrases_for_entry

# --- Main execution ---

# 1. Load the main employment data from 'group_employment.csv'
# Make sure this path is correct for your file location!
try:
    df_employment = pd.read_csv('group_employment.csv', encoding='latin1')
    print("Loaded 'group_employment.csv' successfully.")
except FileNotFoundError:
    print("Error: 'group_employment.csv' not found. Double-check the file name and its location.")
    exit()
except UnicodeDecodeError:
    print("Error: Could not decode 'group_employment.csv' with 'latin1' encoding. Try a different encoding like 'cp1252'.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred loading 'group_employment.csv': {e}")
    exit()

# 2. Load the unique phrases that will form the one-hot encoding columns
# Make sure this path is correct for your file location!
try:
    df_unique_phrases = pd.read_csv('unique_employment_phrases_final_refined_v2.csv')
    unique_phrases = df_unique_phrases['Unique_Employment_Phrases'].tolist()
    print("Loaded unique phrases for one-hot encoding successfully.")
except FileNotFoundError:
    print("Error: 'unique_employment_phrases_final_refined_v2.csv' not found. Cannot perform one-hot encoding without the reference phrases.")
    exit()
except Exception as e:
    print(f"An error occurred loading unique phrases: {e}")
    exit()

# 3. Apply the normalization function to the 'Employment' column of the main DataFrame
df_employment['Normalized_Employment'] = df_employment['Employment'].apply(normalize_employment_string)

# 4. Create an empty DataFrame for one-hot encoding
one_hot_df = pd.DataFrame(0, index=df_employment.index, columns=unique_phrases)

# 5. Populate the one-hot encoded DataFrame
for index, row in df_employment.iterrows():
    for phrase in row['Normalized_Employment']:
        if phrase in one_hot_df.columns:
            one_hot_df.at[index, phrase] = 1

# 6. Concatenate the original DataFrame with the one-hot encoded columns
df_encoded = pd.concat([df_employment, one_hot_df], axis=1)

# 7. Drop the intermediate 'Normalized_Employment' column and the original 'Employment' if desired
df_encoded = df_encoded.drop(columns=['Normalized_Employment', 'Employment'])

print("\nFirst 5 rows of the One-Hot Encoded DataFrame:")
print(df_encoded.head())

# 8. Save the one-hot encoded DataFrame to a CSV file
output_filename = 'group_employment_one_hot_encoded.csv'
df_encoded.to_csv(output_filename, index=False)
print(f"\nOne-Hot Encoded DataFrame saved to '{output_filename}'.")
