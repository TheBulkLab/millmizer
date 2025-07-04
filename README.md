# Millmizer

A Python machine-learning system that predicts optimal milling parameters (ball filling rate, rotation speed, feed rate, milling time) for different milling equipment (Ball Mill A, Attritor Mill X) based on material characteristics (particle size, batch weight, temperature).

## 🚀 Features

- **Synthetic data generation** with realistic distributions  
- **Equipment classification** (RandomForestClassifier)  
- **Parameter regression** (LinearRegression pipelines)  
- **FMEA info** integrated for each parameter  
- **Model persistence** via `joblib`  
- **CLI interface** for user input and report printing  

## 📦 Installation

1. Clone the repository using GitHub Desktop or from a command line:
   git clone https://github.com/yourusername/Millmizer.git
   Open the `Millmizer` folder in your file explorer or terminal.

2. Create a virtual environment:
   python -m venv venv

3. Activate the virtual environment:
   - **Windows (PowerShell):** venv\Scripts\Activate.ps1
   - **Windows (CMD):** venv\Scripts\activate.bat
   - **macOS/Linux:** venv/bin/activate

4. Install the required dependencies:
   pip install -r requirements.txt

## Usage
Run the main script:
python millmizer.py
