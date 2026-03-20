from preprocessing import preprocessing
from training import training
from evaluation import evaluation

print("=== Air Quality Prediction Pipeline ===")

filepath = "data/AirQualityUCI.csv"

# Step 1
data = preprocessing(filepath)

# Step 2 → choose model
model_choice = "random_forest"
# خيارات: "linear", "tree", "random_forest"

results = training(data, model_choice)

# Step 3
evaluation(results)
