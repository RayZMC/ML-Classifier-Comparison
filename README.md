# Machine Learning Comparison

Comparing machine learning techniques for classification and clustering tasks.

## Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install numpy pandas matplotlib scikit-learn scikit-image
```

## Programs

### colour_predict.py

Predicts colour names (red, blue, green, etc.) from RGB values. Compares 6 models:

- Naive Bayes, kNN, Random Forest — each with raw RGB and LAB-converted features

```bash
python src/colour_predict.py data/colour-data.csv
```

### weather_city.py

Predicts which city weather observations came from using supervised classification.

```bash
python src/weather_city.py data/monthly-data-labelled.csv data/monthly-data-unlabelled.csv output.csv
```

### weather_clusters.py

Explores weather data structure using PCA and KMeans clustering.

```bash
python src/weather_clusters.py data/monthly-data-labelled.csv
```

## Data Files

| File                          | Description                                  |
| ----------------------------- | -------------------------------------------- |
| `colour-data.csv`             | RGB values with human-labelled colour names  |
| `monthly-data-labelled.csv`   | Weather data for 26 cities (training data)   |
| `monthly-data-unlabelled.csv` | Weather data with unknown cities (test data) |
| `sample-labels.csv`           | Expected output for unlabelled data          |
