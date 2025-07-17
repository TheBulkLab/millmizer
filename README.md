## Milling Parameter Prediction System

This project provides a command-line tool for predicting optimal milling parameters for industrial processes. Given input material characteristics—specifically particle size, batch weight, and temperature—the system uses a two-stage machine learning pipeline to recommend the best equipment and its operational settings.
The goal is to streamline process setup, improve efficiency, and provide standardized operational guidelines complete with risk analysis.

---

## ✨ Features

- **Predictive Modeling**: Utilizes scikit-learn to forecast optimal process parameters.
- **Two-Stage Prediction**:
    1.  **Classification**: A `RandomForestClassifier` selects the most suitable milling equipment (`Ball_Mill_A` or `Attritor_Mill_X`).
    2.  **Regression**: A set of `LinearRegression` models predicts the specific parameters (ball filling rate, rotation speed, feed rate, milling time) for the chosen equipment.
- **Synthetic Data Generation**: Includes a function to generate realistic training data, allowing the system to be trained from scratch without an initial dataset.
- **Model Persistence**: Automatically saves (using `joblib`) and loads trained models, so training is only required once. A `models/` directory will be created to store them.
- **User-Friendly CLI**: An interactive command-line interface guides the user through entering input parameters.
- **FMEA Integration**: Provides Failure Mode and Effects Analysis (FMEA) information for each parameter, outlining risks, control measures, and acceptable operational ranges.

---

## ⚙️ How it Works

1.  **Save the Code**: Save the provided code as a Python file (e.g., `predictor.py`).

2.  **Run the Script**: Open a terminal or command prompt, navigate to the directory where you saved the file, and run it:
    `python predictor.py`

3.  **First-Time Run**: The first time you run the script, it will detect that no trained models exist. It will automatically generate synthetic data and train all the necessary classifier and regression models. This may take a moment. The trained models will be saved in a new `models/` directory.

4.  **Subsequent Runs**: On all subsequent runs, the script will load the existing models from the `models/` directory for immediate predictions.

5.  **Enter Input Parameters**: The program will prompt you to enter the material and process characteristics.

    ```
    Enter target particle size (5-200 μm): 85
    Enter batch weight (1-500 kg): 150
    Enter process temperature (-10-60°C): 25
    ```

6.  **View the Report**: The system will print a detailed report containing the recommended equipment, confidence level, optimal parameters, and FMEA details.

---

## 🤝 Contributing

Contributions are welcome\! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

## 📄 License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).
