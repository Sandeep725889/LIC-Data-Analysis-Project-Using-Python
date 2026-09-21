#!/usr/bin/env python
# coding: utf-8

# ## project: LIC Insurance Data Analysis using Python
# ## Tools: Python, Pandas,NumPy,Matplotilib,Seaborn,Jupyter Notebook

# # 1. Import Libraries

# In[5]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # 2.Load dataset

# In[6]:


df = pd.read_csv(
    "C:/Users/DELL/Downloads/LIC_Python_Project_Dataset_5000.csv",
    parse_dates=["Policy_Start_Date"]
)

df.head()


# # 3.Dataset Information

# In[7]:


print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

df.info()


# # 4.Basic Statistical Analysis

# In[8]:


df.describe()


# # 5.Check Missing values

# In[9]:


df.isnull().sum()


# # 6.Remove Duplicate Records

# In[10]:


print("Duplicates before:", df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicates after:", df.duplicated().sum())


# # 7.Convert Data Column

# In[11]:


df["Policy_Start_Date"] = pd.to_datetime(
    df["Policy_Start_Date"],
    errors="coerce"
)

df["Year"] = df["Policy_Start_Date"].dt.year
df["Month"] = df["Policy_Start_Date"].dt.month
df["Month_Name"] = df["Policy_Start_Date"].dt.strftime("%b")
print(df.head())
print(df.shape)


# # 8.KPI Analysis

# In[12]:


total_policies = df["Policy_ID"].nunique()

total_premium = df["Premium_Amount"].sum()

average_premium = df["Premium_Amount"].mean()

total_sum_assured = df["Sum_Assured"].sum()

total_claim_amount = df["Claim_Amount"].sum()

claimed_policies = (df["Claim_Status"] == "Claimed").sum()

claim_rate = claimed_policies / total_policies * 100

print("Total Policies:", total_policies)
print("Total Premium:", round(total_premium, 2))
print("Average Premium:", round(average_premium, 2))
print("Total Sum Assured:", round(total_sum_assured, 2))
print("Total Claim Amount:", round(total_claim_amount, 2))
print("Claimed Policies:", claimed_policies)
print("Claim Rate:", round(claim_rate, 2), "%")


# # .CHARTS

# # 9.Policy Count by Policy Type

# In[13]:


policy_count = df["Policy_Type"].value_counts()

plt.figure(figsize=(10,5))

plt.bar(
    policy_count.index,
    policy_count.values
)

plt.title("Policy Count by Policy Type")
plt.xlabel("Policy Type")
plt.ylabel("Number of Policies")
plt.xticks(rotation=35)

plt.tight_layout()
plt.show()


# # 10.Total Premium by Policy Type

# In[60]:


import matplotlib.pyplot as plt
import numpy as np

premium_by_type = (
    df.groupby("Policy_Type")["Premium_Amount"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10,5))

colors = plt.cm.Greens(np.linspace(0.95, 0.4, len(premium_by_type)))

plt.bar(
    premium_by_type.index,
    premium_by_type.values,
    color=colors,
    edgecolor='black'
)

plt.title("Total Premium by Policy Type")
plt.xlabel("Policy Type")
plt.ylabel("Total Premium")
plt.xticks(rotation=35)

plt.tight_layout()
plt.show()


# # Average Premimum by Policy Type

# In[15]:


avg_premium_type = (
    df.groupby("Policy_Type")["Premium_Amount"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10,5))

plt.bar(
    avg_premium_type.index,
    avg_premium_type.values
)

plt.title("Average Premium by Policy Type")
plt.xlabel("Policy Type")
plt.ylabel("Average Premium")
plt.xticks(rotation=35)

plt.tight_layout()
plt.show()


# # Top 10 Cities by Number of Policies

# In[59]:


import matplotlib.pyplot as plt
import numpy as np

city_policy_count = (
    df["City"]
    .value_counts()
    .nlargest(10)
    .sort_values()
)

plt.figure(figsize=(10,6))

colors = plt.cm.Blues(np.linspace(0.4, 0.95, len(city_policy_count)))

plt.barh(
    city_policy_count.index,
    city_policy_count.values,
    color=colors,
    edgecolor='black'
)

plt.title("Top 10 Cities by Number of Policies")
plt.xlabel("Number of Policies")
plt.ylabel("City")

plt.tight_layout()
plt.show()


# # 13.Top 10 Cities by Total premimum

# In[58]:


import matplotlib.pyplot as plt
import numpy as np

city_premium = (
    df.groupby("City")["Premium_Amount"]
    .sum()
    .nlargest(10)
    .sort_values()
)

plt.figure(figsize=(10,6))

# Upper dark, niche light - Gradient
colors = plt.cm.Blues(np.linspace(0.4, 0.95, len(city_premium)))

plt.barh(
    city_premium.index,
    city_premium.values,
    color=colors,
    edgecolor='black'
)

plt.title("Top 10 Cities by Total Premium")
plt.xlabel("Total Premium")
plt.ylabel("City")

plt.tight_layout()
plt.show()


# # 14.Gender Distributition

# In[57]:


import matplotlib.pyplot as plt

gender_count = df["Gender"].value_counts()

plt.figure(figsize=(6,6))

plt.pie(
    gender_count.values,
    labels=gender_count.index,
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops={'edgecolor': 'black'},
    textprops={'fontweight': 'bold'}
)

plt.title("Gender Distribution")
plt.tight_layout()
plt.show()


# # 15.Age group Distribution

# In[19]:


age_count = df["Age_Group"].value_counts()

plt.figure(figsize=(9,5))

plt.bar(
    age_count.index,
    age_count.values
)

plt.title("Age Group Distribution")
plt.xlabel("Age Group")
plt.ylabel("Number of Policies")

plt.tight_layout()
plt.show()


# # 16.Payment Mode Distribution

# In[20]:


payment_count = df["Payment_Mode"].value_counts()

plt.figure(figsize=(9,5))

plt.bar(
    payment_count.index,
    payment_count.values
)

plt.title("Payment Mode Distribution")
plt.xlabel("Payment Mode")
plt.ylabel("Number of Policies")
plt.xticks(rotation=25)

plt.tight_layout()
plt.show()


# # 17.Policy status Distribution

# In[21]:


status_count = df["Policy_Status"].value_counts()

plt.figure(figsize=(8,5))

plt.bar(
    status_count.index,
    status_count.values
)

plt.title("Policy Status Distribution")
plt.xlabel("Policy Status")
plt.ylabel("Number of Policies")

plt.tight_layout()
plt.show()


# # 18. Claim Status Distribution

# In[22]:


claim_count = df["Claim_Status"].value_counts()

plt.figure(figsize=(7,5))

plt.bar(
    claim_count.index,
    claim_count.values
)

plt.title("Claim Status Distribution")
plt.xlabel("Claim Status")
plt.ylabel("Number of Policies")

plt.tight_layout()
plt.show()


# # 19.Claim Rate by Policy Type

# In[23]:


claim_count = df["Claim_Status"].value_counts()

plt.figure(figsize=(7,5))

plt.bar(
    claim_count.index,
    claim_count.values
)

plt.title("Claim Status Distribution")
plt.xlabel("Claim Status")
plt.ylabel("Number of Policies")

plt.tight_layout()
plt.show()


# # 20.Yearly Policy Acquisition Trend

# In[24]:


yearly_policy = (
    df.groupby("Year")["Policy_ID"]
    .count()
)

plt.figure(figsize=(9,5))

plt.plot(
    yearly_policy.index,
    yearly_policy.values,
    marker="o"
)

plt.title("Yearly Policy Acquisition Trend")
plt.xlabel("Year")
plt.ylabel("Number of Policies")

plt.grid()
plt.tight_layout()
plt.show()


# # 21.yearly Premimum Treand

# In[25]:


yearly_premium = (
    df.groupby("Year")["Premium_Amount"]
    .sum()
)

plt.figure(figsize=(9,5))

plt.plot(
    yearly_premium.index,
    yearly_premium.values,
    marker="o"
)

plt.title("Yearly Premium Trend")
plt.xlabel("Year")
plt.ylabel("Total Premium")

plt.grid()
plt.tight_layout()
plt.show()


# # 22.Monthly Premium Trend

# In[26]:


monthly_premium = (
    df.groupby(["Year", "Month"])["Premium_Amount"]
    .sum()
    .reset_index()
)

monthly_premium["Period"] = pd.to_datetime(
    monthly_premium["Year"].astype(str)
    + "-"
    + monthly_premium["Month"].astype(str)
    + "-01"
)

monthly_premium = monthly_premium.sort_values("Period")

plt.figure(figsize=(12,5))

plt.plot(
    monthly_premium["Period"],
    monthly_premium["Premium_Amount"],
    marker="o"
)

plt.title("Monthly Premium Trend")
plt.xlabel("Month")
plt.ylabel("Total Premium")

plt.xticks(rotation=35)
plt.grid()

plt.tight_layout()
plt.show()


# # 23. Top 10Agents by Total Premium

# In[56]:


import matplotlib.pyplot as plt
import seaborn as sns

agent_premium = (
    df.groupby("Agent_ID")["Premium_Amount"]
    .sum()
    .nlargest(10)
    .sort_values()
)

plt.figure(figsize=(10,6))
plt.barh(agent_premium.index.astype(str), agent_premium.values, color=sns.color_palette("viridis", 10))
plt.title("Top 10 Agents by Total Premium")
plt.xlabel("Total Premium")
plt.ylabel("Agent ID")
plt.tight_layout()
plt.show()


# # 24. Premium vs Sum Assured

# In[55]:


import matplotlib.pyplot as plt
import numpy as np

top10 = df.groupby("Agent_ID")["Premium_Amount"].sum().nlargest(10).sort_values()

plt.figure(figsize=(10,6))
plt.barh(top10.index.astype(str), top10.values, color=plt.cm.Blues(np.linspace(0.3, 1, 10)), edgecolor='black')
plt.title("Top 10 Agents by Total Premium - Gradient")
plt.xlabel("Total Premium")
plt.ylabel("Agent ID")
plt.tight_layout()
plt.show()


# # 25.Customer Age vs Premium

# In[54]:


import seaborn as sns
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="Customer_Age", y="Premium_Amount", hue="Policy_Type", palette="Set2", alpha=0.6)
plt.title("Age vs Premium by Policy Type")
plt.show()


# # 26. Premium Distribution

# In[51]:


import seaborn as sns
plt.figure(figsize=(9,5))
sns.histplot(df["Premium_Amount"], bins=30, kde=True, color='orange')
plt.title("Premium Amount Distribution with Trend Line")
plt.show()


# # 27.Premium Distribution by Policy Type Boxplot

# In[33]:


policy_types = df["Policy_Type"].unique()

box_data = [
    df.loc[
        df["Policy_Type"] == policy,
        "Premium_Amount"
    ]
    for policy in policy_types
]

plt.figure(figsize=(11,6))

plt.boxplot(
    box_data,
    labels=policy_types,
    showfliers=False
)

plt.title("Premium Distribution by Policy Type")
plt.xlabel("Policy Type")
plt.ylabel("Premium Amount")

plt.xticks(rotation=35)

plt.tight_layout()
plt.show()


# # 28.Correlation Heatmap

# In[61]:


import matplotlib.pyplot as plt
import numpy as np

numeric_columns = [
    "Customer_Age",
    "Premium_Amount",
    "Sum_Assured",
    "Policy_Term_Years",
    "Claim_Amount"
]

corr = df[numeric_columns].corr()

plt.figure(figsize=(8,6))
plt.imshow(corr, cmap='Blues', vmin=-1, vmax=1)
plt.colorbar(label="Correlation")

plt.xticks(range(len(numeric_columns)), numeric_columns, rotation=35, ha="right")
plt.yticks(range(len(numeric_columns)), numeric_columns)

# text color fix
for i in range(len(numeric_columns)):
    for j in range(len(numeric_columns)):
        value = corr.iloc[i, j]
        color = "white" if abs(value) > 0.5 else "black"
        plt.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontweight="bold")

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# # ANALYSIS & INSIGHTS

# # 29.Most Popular Policy

# In[36]:


most_popular_policy = df["Policy_Type"].value_counts().idxmax()

print(
    "Most Popular Policy Type:",
    most_popular_policy
)


# # 30. Highest Premium Policy

# In[37]:


highest_premium_policy = (
    df.groupby("Policy_Type")["Premium_Amount"]
    .sum()
    .idxmax()
)

print(
    "Policy Type with Highest Total Premium:",
    highest_premium_policy
)


# # 30. Highest Average Premium

# In[38]:


highest_avg_premium = (
    df.groupby("Policy_Type")["Premium_Amount"]
    .mean()
    .idxmax()
)

print(
    "Highest Average Premium Policy:",
    highest_avg_premium
)


# # 32.Top City 

# In[39]:


top_city = (
    df.groupby("City")["Premium_Amount"]
    .sum()
    .idxmax()
)

print("Top City by Premium:", top_city)


# # 33.Top Agent

# In[40]:


top_agent = (
    df.groupby("Agent_ID")["Premium_Amount"]
    .sum()
    .idxmax()
)

print("Top Agent:", top_agent)


# # 34.Most common Payment mode

# In[41]:


most_common_payment = (
    df["Payment_Mode"]
    .value_counts()
    .idxmax()
)

print(
    "Most Common Payment Mode:",
    most_common_payment
)


# # 35.Most Common Policy Status

# In[42]:


most_common_status = (
    df["Policy_Status"]
    .value_counts()
    .idxmax()
)

print(
    "Most Common Policy Status:",
    most_common_status
)


# # 36.Highest claim Rate Policy

# In[43]:


claim_rate = (
    df.assign(
        Claimed=df["Claim_Status"] == "Claimed"
    )
    .groupby("Policy_Type")["Claimed"]
    .mean()
    .mul(100)
)

highest_claim_policy = claim_rate.idxmax()

print(
    "Highest Claim Rate Policy:",
    highest_claim_policy
)

print(
    "Claim Rate:",
    round(claim_rate.max(), 2),
    "%"
)


# # 37.Gender With Highest Policy Count

# In[44]:


top_gender = df["Gender"].value_counts().idxmax()

print(
    "Gender with Highest Policy Count:",
    top_gender
)


# # 38.Hisghest Premium Year

# In[45]:


highest_premium_year = (
    df.groupby("Year")["Premium_Amount"]
    .sum()
    .idxmax()
)

print(
    "Highest Premium Year:",
    highest_premium_year
)


# # 39.Highest Policy Acquisition Year

# In[46]:


highest_policy_year = (
    df.groupby("Year")["Policy_ID"]
    .count()
    .idxmax()
)

print(
    "Highest Policy Acquisition Year:",
    highest_policy_year
)


# # FINAL INSIGHTS TABLE

# In[47]:


insights = {
    "Metric": [
        "Total Policies",
        "Total Premium",
        "Average Premium",
        "Total Sum Assured",
        "Total Claim Amount",
        "Claim Rate",
        "Most Popular Policy",
        "Top City",
        "Top Agent",
        "Most Common Payment Mode",
        "Most Common Policy Status",
        "Highest Claim Rate Policy"
    ],
    
    "Value": [
        total_policies,
        round(total_premium, 2),
        round(average_premium, 2),
        round(total_sum_assured, 2),
        round(total_claim_amount, 2),
        round(claim_rate, 2),
        most_popular_policy,
        top_city,
        top_agent,
        most_common_payment,
        most_common_status,
        highest_claim_policy
    ]
}

insights_df = pd.DataFrame(insights)

insights_df


# # EXPORT ANALYSIS RESULT

# In[49]:


insights_df.to_excel(
    "LIC_Project_Insights.xlsx",
    index=False
)


# # FINAL PROJECT CONCLUSION

# In[50]:


print("========== LIC PYTHON DATA ANALYST PROJECT ==========")

print("Total Policies:", total_policies)

print("Total Premium:", round(total_premium, 2))

print("Average Premium:", round(average_premium, 2))

print("Claim Rate:", round(claim_rate, 2), "%")

print("Most Popular Policy:", most_popular_policy)

print("Top City:", top_city)

print("Top Agent:", top_agent)

print("Highest Claim Rate Policy:", highest_claim_policy)

print("======================================================")


# In[ ]:




