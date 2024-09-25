import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Load the Titanic dataset into DataFrame
df = sns.load_dataset('titanic')

# Inspect first rows of the DataFrame
print(df.head())

print(df.head(10))
print(df.shape)
print(df.dtypes)

print(df['survived'].value_counts())

# Print original columns
print("Original columns:")
print(df.columns)

# Rename columns
new_column_names = {
    'pclass': 'Passenger Class',
    'sex': 'Sex',
    'age': 'Age',
    'sibsp': 'Siblings/Spouses Aboard',
    'parch': 'Parents/Children Aboard',
    'fare': 'Fare Amount',
    'survived': 'Survival Status'
}
df.rename(columns=new_column_names, inplace=True)

# Print updated columns
print("\nUpdated columns:")
print(df.columns)

df.rename(columns=new_column_names, inplace=True)

print(df.describe())

# Inspect value counts for all categorical columns
for col in df.select_dtypes(include=['object', 'category']).columns:
    print(f'Value counts for column: {col}')
    print(df[col].value_counts())
    print()
    # Display count plot
    sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.show()

# Show all plots
plt.show()

print(df[['Sex', 'Survival Status']].value_counts())

# Calculate survival rates by class and sex
survival_counts = df.groupby(['Passenger Class', 'Sex', 'Survival Status']).size().unstack(fill_value=0)
print(survival_counts)

# Calculate the total count of passengers in each group
total_counts = survival_counts.sum(axis=1)

# Calculate survival proportions
survival_proportions = survival_counts.div(total_counts, axis=0)
print(survival_proportions)

# Create a stacked bar plot
ax = survival_proportions.plot(kind='bar', stacked=True, color=['red', 'green'], figsize=(10, 6))

# Customize the plot
plt.title('Survival Rate by Passenger Class and Sex')
plt.xlabel('Passenger Class')
plt.ylabel('Proportion of Passengers')
plt.xticks(rotation=0)

# Set the legend with proper labels
plt.legend(title='Survived', labels=['No', 'Yes'], loc='upper right')

plt.show()





'''# Create a new column to identify mothers
df['is_mother'] = (df['Sex'] == 'female') & (df['Parents/Children Aboard'] > 0)
df['is_woman_without_children'] = (df['Sex'] == 'female') & (df['Parents/Children Aboard'] == 0)

# Calculate survival rates
mothers_survival_rate = df[df['is_mother']]['Survival Status'].mean()
women_without_children_survival_rate = df[df['is_woman_without_children']]['Survival Status'].mean()

# Create a DataFrame for the survival statistics
survival_stats = pd.DataFrame({
    'Group': ['Mothers', 'Women without Children'],
    'Survival Rate': [mothers_survival_rate, women_without_children_survival_rate]
})

# Create a bar plot for survival rates
plt.figure(figsize=(8, 6))
sns.barplot(x='Group', y='Survival Rate', data=survival_stats, palette='pastel')

# Customize the plot
plt.title('Survival Rate: Mothers vs. Women Without Children')
plt.xlabel('Group')
plt.ylabel('Survival Rate')

# Add percentage labels on top of the bars
for index, value in enumerate(survival_stats['Survival Rate']):
    plt.text(index, value + 0.02, f'{value:.1%}', ha='center')

plt.ylim(0, 1)  # Set y-axis limit to 1
plt.show()

# Plotting the scatter plot of Age vs. Fare Amount
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='Fare Amount', hue='Survival Status', style='Sex', palette='deep', alpha=0.7)

# Customize the plot
plt.title('Scatter Plot of Age vs. Fare on the Titanic')
plt.xlabel('Age')
plt.ylabel('Fare Amount')
plt.legend(title='Survival Status')
plt.grid()
plt.show()

# Calculate and visualize survival rates by class and sex
survival_counts = df.groupby(['Passenger Class', 'Sex', 'Survival Status']).size().unstack(fill_value=0)
survival_counts = survival_counts.div(survival_counts.sum(axis=1), axis=0)

# Create a stacked bar plot
ax = survival_counts.plot(kind='bar', stacked=True, color=['red', 'green'], figsize=(10, 6))

# Customize the plot
plt.title('Survival Rate by Passenger Class and Sex')
plt.xlabel('Passenger Class')
plt.ylabel('Proportion of Passengers')
plt.xticks(rotation=0)

# Set the legend with proper labels
plt.legend(title='Survived', labels=['No', 'Yes'], loc='upper right')

# Add custom labels for 'Sex' on the x-tick labels
ax.set_xticklabels(['1st Class\n(Male)', '1st Class\n(Female)', 
                    '2nd Class\n(Male)', '2nd Class\n(Female)', 
                    '3rd Class\n(Male)', '3rd Class\n(Female)'], rotation=0)

plt.show()'''