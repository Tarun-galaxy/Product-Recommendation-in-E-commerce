from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("product_realistic_unique_names.csv")

# Keep full copy for UI
display_df = df.copy()

# Remove leakage columns from training
df = df.drop(columns=["User_ID", "Product_ID", "Cart_Added", "Rating", "Image_URL"], errors="ignore")

# ==============================
# ENCODING
# ==============================
categorical_columns = [
    "Gender", "Location", "Product_Name",
    "Brand", "Product_Type", "Category",
    "Search_Query"
]

label_encoders = {}

for col in categorical_columns:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# ==============================
# TRAIN MODEL
# ==============================
X = df.drop("Bought", axis=1)
y = df["Bought"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# ROUTE
# ==============================
@app.route("/", methods=["GET", "POST"])
def index():
    probability = None
    prediction = None
    recommendations = []

    selected_gender = None
    selected_location = None
    selected_query = None

    if request.method == "POST":

        selected_gender = request.form.get("gender")
        selected_location = request.form.get("location")
        selected_query = request.form.get("search_query")

        # Prepare input for prediction
        input_df = pd.DataFrame(columns=X.columns)

        for col in X.columns:
            input_df.loc[0, col] = X[col].mean()

        input_df["Gender"] = label_encoders["Gender"].transform([selected_gender])[0]
        input_df["Location"] = label_encoders["Location"].transform([selected_location])[0]
        input_df["Search_Query"] = label_encoders["Search_Query"].transform([selected_query])[0]

        input_df = input_df[X.columns]

        prediction = model.predict(input_df)[0]
        probability = round(model.predict_proba(input_df)[0][1] * 100, 2)

        # Recommendations (Rating based)
        filtered_products = display_df[display_df["Search_Query"] == selected_query]
        recommendations = filtered_products.sort_values(
            by="Rating", ascending=False
        ).head(5).to_dict(orient="records")

    return render_template(
        "index.html",
        genders=label_encoders["Gender"].classes_,
        locations=label_encoders["Location"].classes_,
        queries=label_encoders["Search_Query"].classes_,
        probability=probability,
        prediction=prediction,
        recommendations=recommendations,
        selected_gender=selected_gender,
        selected_location=selected_location,
        selected_query=selected_query
    )


if __name__ == "__main__":
    app.run(debug=True)
