# Tomato Leaf Check

Photograph one tomato leaf. The app names the most likely of 7 problems (early blight, late blight, spotted wilt, leaf miner, magnesium deficiency, nitrogen deficiency, or healthy), shows where the model looked (Grad-CAM), and gives advice on what to do. When the model is not confident, it says "Not sure" and shows its two closest matches instead of guessing.

Mini project of Group 23, B.Tech AI/ML, Universal AI University: *AI-Based Early Detection and Management System for Crop Diseases and Pest Infestations*. Guide: Mr. Rajesh Bhise.

## Model

A frozen MobileNetV2 (ImageNet weights) turns each photo into 1,280 features, and a logistic-regression classifier picks the class. It was trained on a leak-free, cleaned split of the Tomato-Village dataset (single leaves on paper) plus about 1,100 leaves cut from Tomato-Village field photos. The "Not sure" threshold was chosen so that answered photos are right about 80% of the time. On real field photos from other sources the model is often wrong, and the app says so.

## Run on your computer

```
pip install -r requirements.txt     # on a Mac, replace tensorflow-cpu with tensorflow
streamlit run app.py
```

The `models` folder must contain `final_model.keras` and `final_model_info.json`.

## Credits

- Data: Gehlot, M., Saxena, R.K. and Gandhi, G.C. "Tomato-Village": a dataset for end-to-end tomato disease detection in a real-world environment. *Multimedia Systems* (2023). https://doi.org/10.1007/s00530-023-01158-y
- PlantDoc (Singh et al., 2020), CC BY 4.0, for the field sample photo.
- Advice: TNAU Agritech Portal (Crop Production Guide, Horticulture 2020, updated 2023) and the AESA-based IPM package for tomato (NIPHM, NCIPM and Directorate of Plant Protection, Government of India). Always confirm sprays with your nearest Krishi Vigyan Kendra.
