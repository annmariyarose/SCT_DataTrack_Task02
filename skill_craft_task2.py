import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\USER\Downloads\train (1).csv")


# Handling Missing Values (Safe for pandas 3.0+)

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop(columns=['Cabin'])  # Dropping column with too many nulls

# Set visual style
sns.set(style="whitegrid")


# Visualization 1: Survival Count

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Survived', hue='Survived', palette={0: 'yellow', 1: 'blue'}, legend=False)
plt.title('Survival Count')
plt.xticks([0, 1], ['Not Survived', 'Survived'])
plt.tight_layout()
plt.savefig("survival_count.png")
plt.show()
plt.close()


# Visualization 2: Survival by Gender

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Sex', hue='Survived', palette={0: 'yellow', 1: 'blue'})
plt.title('Survival by Gender')
plt.tight_layout()
plt.savefig("survival_by_gender.png")
plt.show()
plt.close()


# Visualization 3: Survival by Class

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Pclass', hue='Survived', palette={0: 'pink', 1: 'blue'})
plt.title('Survival by Passenger Class')
plt.tight_layout()
plt.savefig("survival_by_class.png")
plt.show()
plt.close()

# Correlation Heatmap

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()
plt.close()


# Show First Few Rows of Cleaned Dataset

print("\n Preview of Cleaned DataFrame:\n")
print(df.head(10))  # Preview first 10 rows


