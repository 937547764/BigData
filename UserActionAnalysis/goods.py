import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Load the dataset
file_path = 'hdfs://namenode:9000//user_log_format1.csv'

user_data = pd.read_csv(file_path)
# Calculating the cmatplotlibount of purchases for each category
purchase_counts_cat = user_data[user_data['action_type'] == 2].groupby('cat_id').size()

# Converting to a DataFrame
df_purchase_counts_cat = pd.DataFrame(purchase_counts_cat, columns=['purchase_count_cat'])

# Displaying the descriptive statistics
df_purchase_counts_cat_describe = df_purchase_counts_cat.describe()
print(df_purchase_counts_cat_describe)

# Calculating the top 10 categories by purchase count
top_10_categories = purchase_counts_cat.nlargest(10)

# Converting to DataFrame for plotting
df_top_10_categories = pd.DataFrame(top_10_categories, columns=['purchase_count_cat']).reset_index()
# Plotting a pie chart_cat
plt.figure(figsize=(10, 8))
plt.pie(df_top_10_categories['purchase_count_cat'], labels=df_top_10_categories['cat_id'], autopct='%1.1f%%', startangle=140,
        colors=["#719AC0", "#7CA9D3", "#88BAE8", "#96CDFF", "#B7D7FF",
                "#D8E1FF", "#DACEEE", "#DBBADD", "#CDA6C0", "#BE92A2"])
plt.title('Top 10 Categories by Sales Volume')
plt.legend(df_top_10_categories['cat_id'], title='Category ID', loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.savefig('Top10Categories.png')


purchase_counts_item = user_data[user_data['action_type'] == 2].groupby('item_id').size()

# Converting to a DataFrame
df_purchase_counts_item = pd.DataFrame(purchase_counts_item, columns=['purchase_counts_item'])

# Displaying the descriptive statistics
df_purchase_counts_item_describe = df_purchase_counts_item.describe()
print(df_purchase_counts_item_describe)

# Calculating the top 10 categories by purchase count
top_10_categories = purchase_counts_item.nlargest(10)

# Converting to DataFrame for plotting
df_top_10_categories = pd.DataFrame(top_10_categories, columns=['purchase_counts_item']).reset_index()
# Plotting a pie chart_cat
plt.figure(figsize=(10, 8))
plt.pie(df_top_10_categories['purchase_counts_item'], labels=df_top_10_categories['item_id'], autopct='%1.1f%%', startangle=140,
        colors=["#719AC0", "#7CA9D3", "#88BAE8", "#96CDFF", "#B7D7FF",
                "#D8E1FF", "#DACEEE", "#DBBADD", "#CDA6C0", "#BE92A2"])
plt.title('Top 10 items by Sales Volume')
plt.legend(df_top_10_categories['item_id'], title='Item ID', loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.savefig('Top10items.png')

