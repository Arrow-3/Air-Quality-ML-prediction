from preprocessing import preprocessing
from exploratory_analysis import exploratory_analysis
from training import training
from evaluation import evaluation

def main():

    print("=== Air Quality Prediction Pipeline ===")

    filepath = "data/AirQualityUCI.csv"

    # Step 1: Preprocessing
    data = preprocessing(filepath)

    # Step 2: EDA
    exploratory_analysis(data)

    # Step 3: Model selection
    model_choice = "random_forest"

    # Step 4: Training
    results = training(data, model_choice)

    # Step 5: Evaluation
    evaluation(results)


if __name__ == "__main__":
    main()
