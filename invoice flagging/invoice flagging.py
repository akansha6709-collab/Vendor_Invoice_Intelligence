import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

conn = sqlite3.connect("Data/inventory.db")

purchases = pd.read_sql_query("SELECT * FROM purchases", conn)
purchase_prices = pd.read_sql_query("SELECT * FROM purchase_prices", conn)
vendor_invoice = pd.read_sql_query("SELECT * FROM vendor_invoice", conn)
begin_inventory = pd.read_sql_query("SELECT * FROM begin_inventory", conn)
end_inventory = pd.read_sql_query("SELECT * FROM end_inventory", conn)

# print(purchases.head())
# print(purchase_prices.head())
# print(vendor_invoice.head())
# print(begin_inventory.head())
# print(end_inventory.head())


purchase_agg_df = pd.read_sql_query("""
    SELECT
        p.PONumber,
        COUNT(DISTINCT p.Brand) AS total_brands,
        SUM(p.Quantity) AS total_item_quantity,
        SUM(p.Dollars) AS total_item_dollars,
        AVG(julianday(p.ReceivingDate) - julianday(p.PODate))
            AS avg_receiving_delay
    FROM purchases p
    GROUP BY p.PONumber
    LIMIT 10
""", conn)

# print(purchase_agg_df)

invoice_df = pd.read_sql_query("""
SELECT
    vi.Quantity AS invoice_quantity,
    vi.Dollars AS invoice_dollars,
    vi.Freight,
    (julianday(vi.InvoiceDate) - julianday(vi.PODate))
        AS days_po_to_invoice,
    (julianday(vi.PayDate) - julianday(vi.InvoiceDate))
        AS days_to_pay
FROM vendor_invoice vi
""", conn)

# print(invoice_df.head())
# print(invoice_df.shape)


df = pd.read_sql_query("""
WITH purchase_agg AS (
    SELECT
        p.PONumber,
        COUNT(DISTINCT p.Brand) AS total_brands,
        SUM(p.Quantity) AS total_item_quantity,
        SUM(p.Dollars) AS total_item_dollars,
        AVG(julianday(p.ReceivingDate) - julianday(p.PODate))
            AS avg_receiving_delay
    FROM purchases p
    GROUP BY p.PONumber
)

SELECT
    vi.PONumber,
    vi.Quantity AS invoice_quantity,
    vi.Dollars AS invoice_dollars,
    vi.Freight,
    (julianday(vi.InvoiceDate) - julianday(vi.PODate))
        AS days_po_to_invoice,
    (julianday(vi.PayDate) - julianday(vi.InvoiceDate))
        AS days_to_pay,
    pa.total_brands,
    pa.total_item_quantity,
    pa.total_item_dollars,
    pa.avg_receiving_delay

FROM vendor_invoice vi
LEFT JOIN purchase_agg pa
    ON vi.PONumber = pa.PONumber
""", conn)

# print(df.head())
# print(df.shape)

def create_invoice_risk_label(row):

    # Invoice total mismatch with item-level total
    if (abs(row["invoice_dollars"] - row["total_item_dollars"]) > 5):
        return 1

    # Abnormally high receiving delay
    if row["avg_receiving_delay"] > 10:
        return 1

    return 0

df["flag_invoice"] = df.apply(create_invoice_risk_label, axis=1)
df["flag_invoice"].value_counts()

# print("Invoice Risk Flag Counts:")
# print(df["flag_invoice"].value_counts())

df["flag_invoice"].value_counts().plot(kind="bar")
# plt.show()

plt.figure(figsize=(20, 10))
sns.heatmap(df.iloc[:, :-1].corr(), annot=True)
plt.show()


flagged = df[df['flag_invoice'] == 1]
normal = df[df['flag_invoice'] == 0]

# print("Flagged invoices:", flagged.shape)
# print("Normal invoices:", normal.shape)
# print(flagged.head())
# print(normal.head())


metrics = [
    'invoice_quantity',
    'invoice_dollars',
    'Freight',
    'days_po_to_invoice',
    'days_to_pay',
    'total_brands'
]

significant_features = []
non_significant_features = []
results = []


from scipy.stats import ttest_ind

for metric in metrics:
    flagged_mean = flagged[metric].mean()
    normal_mean = normal[metric].mean()

    t_stat, p_value = ttest_ind(
        flagged[metric].dropna(),
        normal[metric].dropna(),
        equal_var=False
    )

    if p_value < 0.05:
        significant_features.append(metric)
        results.append({
            "metric": metric,
            "flagged_mean": flagged_mean.round(2),
            "normal_mean": normal_mean.round(2),
            "p_value": p_value.round(3)
        })
    else:
        non_significant_features.append(metric)

# print(df.columns.tolist())        
# print("Metrics tested:", metrics)
# print("Significant:", significant_features)
# print("Non-significant:", non_significant_features)
# print(df.columns.tolist())

X = df[['invoice_quantity', 'invoice_dollars', 'Freight',
        'total_brands', 'total_item_quantity',
        'days_po_to_invoice', 'total_item_dollars']]

y = df['flag_invoice']

print(X.describe().round())

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()

from sklearn.model_selection import train_test_split

X = df[['invoice_quantity', 'invoice_dollars', 'Freight',
        'total_brands', 'total_item_quantity',
        'days_po_to_invoice', 'total_item_dollars']]

y = df['flag_invoice']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Model 1: Logistic Regression
model1 = LogisticRegression(random_state=42)
model1.fit(X_train_scaled, y_train)

# Model 2: Decision Tree Classifier
model2 = DecisionTreeClassifier(random_state=42)
model2.fit(X_train_scaled, y_train)

# Model 3: Random Forest Classifier
model3 = RandomForestClassifier(random_state=42)
model3.fit(X_train_scaled, y_train)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Logistic Regression
y_pred1 = model1.predict(X_test_scaled)

print("Logistic Regression Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred1))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred1))
print("Classification Report:\n", classification_report(y_test, y_pred1))


# Decision Tree Classifier
y_pred2 = model2.predict(X_test_scaled)

print("\nDecision Tree Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred2))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred2))
print("Classification Report:\n", classification_report(y_test, y_pred2))


# Random Forest Classifier
y_pred3 = model3.predict(X_test_scaled)

print("\nRandom Forest Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred3))
print("Classification Report:\n", classification_report(y_test, y_pred3))

