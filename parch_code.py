'''This code was originally meant to see of "women traveling with
 children (mothers)" were more likely to survive than women traveling
   without children.  My hypothesis was that mothers had a better 
   chance at survival.  This is also what Google and a peer 
   reviewed publication says.  So, when my results turned out 
   different, I had to dig deeper.  it turns out that with the 
   dataset in Seaborn I don't think it is possible to tell this. 
     The parch column is "parents and children traveling together".
         From my understanding, you don't know whether this is a 25 
         year old woman with children or a 25 year old women traveling
           with her parents (in which case she may not have kids but 
           would still show up at >0 for  the parch column).  So, this
             was excluded from my analysis.'''


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Load the Titanic dataset
df = sns.load_dataset('titanic')

# Convert categorical variables to numeric if necessary
df['sex'] = df['sex'].map({'male': 1, 'female': 0})

# Create a new column to identify mothers
df['is_mother'] = (df['sex'] == 0) & (df['parch'] > 0)

# Create a new column to identify women without children
df['is_woman_without_children'] = (df['sex'] == 0) & (df['parch'] == 0)

# Check counts of each group
num_mothers = df['is_mother'].sum()
num_women_without_children = df['is_woman_without_children'].sum()
print(f"Number of Mothers: {num_mothers}, Number of Women Without Children: {num_women_without_children}")

# Calculate survival rates
mothers_survival_rate = df[df['is_mother']]['survived'].mean()
women_without_children_survival_rate = df[df['is_woman_without_children']]['survived'].mean()

# Print survival rates
print(f"Survival Rate for Mothers: {mothers_survival_rate}")
print(f"Survival Rate for Women Without Children: {women_without_children_survival_rate}")

# Create a DataFrame for the survival statistics
survival_stats = pd.DataFrame({
    'Group': ['Mothers', 'Women without Children'],
    'Survival Rate': [mothers_survival_rate, women_without_children_survival_rate]
})

# Create a bar plot if we have valid data
if survival_stats['Survival Rate'].notnull().any():
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Group', hue='Group', y='Survival Rate', data=survival_stats, palette='pastel')

    # Customize the plot
    plt.title('Survival Rate: Mothers vs. Women Without Children')
    plt.xlabel('Group')
    plt.ylabel('Survival Rate')

    # Add percentage labels on top of the bars
    for index, value in enumerate(survival_stats['Survival Rate']):
        plt.text(index, value + 0.02, f'{value:.1%}', ha='center')

    plt.ylim(0, 1)  # Set y-axis limit to 1
    plt.show()
else:
    print("No valid survival rates to plot.")