# Millmizer

This project is a machine learning system designed to predict optimal milling parameters based on material characteristics. It recommends the most suitable milling equipment and the ideal operational parameters (ball filling rate, rotation speed, feed rate, and milling time) to achieve a desired outcome.

The system is built as a proof-of-concept and uses synthetically generated data to train its models.

## ⚙️ Features

-   **Equipment Recommendation**: Classifies the best milling equipment (`Ball_Mill_A` or `Attritor_Mill_X`) based on input features.
-   **Parameter Optimization**: Predicts four key milling parameters using regression models tailored to each equipment type.
-   **Automated Model Training**: If predictive models are not found, the script will automatically generate synthetic data and train new models.
-   **Data-Driven Predictions**: Utilizes a Random Forest Classifier for equipment selection and Linear Regression for parameter prediction.
-   **Failure Mode and Effects Analysis (FMEA)**: Provides risk and control information for each predicted parameter to guide operations.
-   **Command-Line Interface**: Simple and interactive CLI to input material data and receive a full prediction report.

## 🔧 How It Works

The system operates in two main phases: Model Training and Prediction.

1.  **Model Training** (`train_models`):
    -   If no pre-trained models are found in the `models/` directory, this phase is initiated.
    -   **Synthetic Data Generation**: Creates a realistic dataset by simulating material properties (particle size, batch weight, temperature) and corresponding equipment choices and parameters.
    -   **Classification**: A `RandomForestClassifier` is trained to decide which mill to use (`Ball_Mill_A` or `Attritor_Mill_X`).
    -   **Regression**: For each piece of equipment, separate `LinearRegression` models are trained to predict the optimal settings for `ball_filling_rate`, `rotation_speed`, `feed_rate`, and `milling_time`.
    -   **Serialization**: All trained models (classifier, scaler, label encoder, and regression models) are saved to the `models/` directory using `joblib`.

2.  **Prediction** (`predict_parameters`):
    -   The system loads the pre-trained models from the `models/` directory.
    -   The user provides three inputs: target particle size (μm), batch weight (kg), and process temperature (°C).
    -   The inputs are scaled using the saved `StandardScaler`.
    -   The classifier predicts the best equipment and the confidence of this prediction.
    -   Based on the chosen equipment, the corresponding regression models predict the four key milling parameters.
    -   A comprehensive report is displayed, including the recommendations, confidence scores, and FMEA information.

## 🚀 Getting Started

### Prerequisites

-   Python 3.6 or newer.

### Installation

1.  **Clone the repository or download the source code.**

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

Execute the main script from your terminal:

```bash
python millmizer.py
```

First Run: The script will detect that no models are present, automatically train them, and save them in a newly created models/ directory.
Subsequent Runs: The script will load the existing models to make predictions instantly.

You will be prompted to enter the material characteristics. Provide the values within the specified ranges to get a prediction.

### Example Interaction:
```bash
Milling Parameter Prediction System

==================================================
Loading existing models...
All models loaded successfully!

==================================================
PARAMETER INPUT
==================================================
Enter target particle size (5-200 μm): 75
Enter batch weight (1-500 kg): 120
Enter process temperature (-10-60°C): 25
This will generate a detailed report with the recommended equipment and parameters.
```

### 📁 Project Structure
```
├── millmizer.py            # Main Python script
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
├── CONTRIBUTING.md         # Project documentation
├── CODE_OF_CONDUCT.md      # Project documentation
└── LICENSE                 # MIT License
```

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Industrial process optimization research community
- FMEA methodology standards
- Open source machine learning libraries (scikit-learn, pandas, numpy)

## 📞 Support

- 📖 Documentation: [Wiki](https://github.com/r0bin-kim/dismizer/wiki)
- 🐛 Bug Reports: [Issues](https://github.com/r0bin-kim/dismizer/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/r0bin-kim/dismizer/discussions)

