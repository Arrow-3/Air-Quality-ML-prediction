import matplotlib.pyplot as plt

def evaluation(results):

    print("Model:", results["model"])
    print("RMSE:", results["rmse"])

    plt.figure()
    plt.plot(results["y_test"].values, label="Actual")
    plt.plot(results["y_pred"], label="Predicted")
    plt.legend()
    plt.title(f"Model: {results['model']}")
    plt.xlabel("Time Step")
    plt.ylabel("RH")

    plt.savefig("results_plot.png")
    plt.show()
