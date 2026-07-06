# Multimodal ML – Housing Price Prediction Using Images + Tabular Data

Predict housing prices by fusing structured (tabular) data with visual features extracted from house images using a CNN.

## 🎯 Objective
Housing prices depend on both measurable attributes (size, location, bedrooms) and visual quality/condition that numbers alone can't capture. This project builds a multimodal model that combines CNN-extracted image features with tabular data to produce more accurate price predictions than either modality alone.

## 🗂️ Dataset
- **Tabular data:** Housing Sales Dataset (e.g., Kaggle "House Sales in King County, USA" — https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)
- **Image data:** Custom/paired house images (e.g., Kaggle "Houses Dataset" by Ahmed and Moustafa — https://github.com/emanhamed/Houses-dataset), matched to the same properties

## 🛠️ Methodology / Approach
1. **Data Loading & Alignment** – Load the tabular dataset and match each row to its corresponding house image(s).
2. **Image Preprocessing** – Resize/normalize images; augment training images (flips, rotations) to reduce overfitting.
3. **CNN Feature Extraction** – Use a CNN (custom small CNN or a pretrained model like ResNet/MobileNet with the top removed) to extract a fixed-length feature vector from each house image.
4. **Feature Fusion** – Concatenate the CNN image feature vector with the preprocessed/scaled tabular features into a single combined feature vector.
5. **Model Training** – Train a regression head (dense neural network or gradient boosting model) on the fused features to predict price.
6. **Evaluation** – Evaluate using MAE and RMSE, and compare against tabular-only and image-only baselines to quantify the benefit of fusion.

## 📊 Key Results / Observations
- The fused (image + tabular) model achieved lower MAE/RMSE than either the tabular-only or image-only baseline.
- Image features contributed most for properties where visual condition/style significantly affects price (e.g., renovated vs outdated interiors).
- Tabular features (square footage, location, bedrooms) remained the dominant predictors overall, with images providing a smaller but consistent accuracy boost.

*(Replace with your actual MAE/RMSE numbers and baseline comparisons after training.)*

## ⚙️ Tools & Technologies
- Python 3, Jupyter Notebook
- TensorFlow/Keras or PyTorch (CNN + fusion model)
- pandas, numpy, scikit-learn (tabular preprocessing, metrics)
- matplotlib/seaborn (visualizations)

## 🚀 How to Run
```bash
git clone https://github.com/<your-username>/multimodal-housing-price-prediction.git
cd multimodal-housing-price-prediction
pip install -r requirements.txt
jupyter notebook Multimodal_Housing_Price_Prediction.ipynb
```

## 📁 Files
| File | Description |
|---|---|
| `Task3_Phase2.ipynb` | Full pipeline: CNN feature extraction, fusion, training, evaluation |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

## 📌 Skills Demonstrated
- Multimodal machine learning
- Convolutional Neural Networks (CNNs)
- Feature fusion (image + tabular)
- Regression modeling and evaluation (MAE, RMSE)
