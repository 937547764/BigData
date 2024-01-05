import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'hdfs://namenode:9000//user_log_format1.csv'
df = pd.read_csv(file_path)
print(len(df))
print( df['action_type'].value_counts())


action_counts = df['action_type'].value_counts(normalize=True) * 100


plot_data = action_counts.reset_index()
plot_data.columns = ['Action Type', 'Percentage']
plot_data['Action Type'] = plot_data['Action Type'].map({0: 'Clicked', 1: 'Added to Cart', 3: 'Added to Favorites', 2: 'Purchased'})


ax = sns.barplot(x='Action Type', y='Percentage', data=plot_data, palette='Blues')
plt.title('Percentage of Each User Action')
plt.ylabel('Percentage (%)')
plt.xlabel('User Action')
ax.set_ylim(0, plot_data['Percentage'].max() + 10) 

for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points')


plt.savefig('ActionPercentage.png')


df = df.sort_values(by=['user_id', 'time_stamp'])
grouped = df.groupby(['user_id', 'item_id'])['action_type'].apply(list)
total = len(grouped)
print(total)


def check_sequence(user_data, sequence):
    for action in sequence:
        if action not in user_data:
            return False
    return True



def calculate_conversion(df, sequence):
    matches = grouped.apply(lambda x: check_sequence(x, sequence))
    matched = matches.sum()


    if total == 0:
        return 0
    else:
        return matched


sequences = {
    "Click to Purchase": [0, 2],
    "Click, Add to Cart, Purchase": [0, 1, 2],
    "Favorites to Purchase": [3, 2],
    "Favorites, Add to Cart, Purchase": [3, 1]
}


conversion = {name: calculate_conversion(df, seq) for name, seq in sequences.items()}

for name, num in conversion.items():
    print(f"{name} num: {num} ")


