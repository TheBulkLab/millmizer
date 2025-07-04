# MillingParameterPredictor

A Python machine-learning system that predicts optimal milling parameters (ball filling rate, rotation speed, feed rate, milling time) for different milling equipment (Ball Mill A, Attritor Mill X) based on material characteristics (particle size, batch weight, temperature).

## 🚀 Features

- **Synthetic data generation** with realistic distributions  
- **Equipment classification** (RandomForestClassifier)  
- **Parameter regression** (LinearRegression pipelines)  
- **FMEA info** integrated for each parameter  
- **Model persistence** via `joblib`  
- **CLI interface** for user input and report printing  

## 📦 Installation

1. Clone the repo  
   ```bash
   git clone https://github.com/yourusername/MillingParameterPredictor.git
   cd MillingParameterPredictor
