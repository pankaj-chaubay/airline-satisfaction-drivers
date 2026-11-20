# Data

This project uses the Kaggle data set **Airlines Customer satisfaction** (file `Invistico_Airline.csv`, 129,880 rows, 22 columns). It is a passenger satisfaction survey from an airline whose name was changed to "Invistico". It is not US government (BTS) data.

The file is not included in this repository. To run the full analysis:

1. Download it from Kaggle (search for "Airlines Customer satisfaction", file `Invistico_Airline.csv`).
2. Save it as `data/Invistico_Airline.csv`.
3. Run `python -m airline_satisfaction`.

The loader accepts the label column as `satisfaction` (Kaggle) or `customer_satisfaction` (the name in my original coursework file). It checks that all 21 feature columns are present.

Without the file, `python -m airline_satisfaction --sample` runs the same steps on generated data with the same columns. The tests and CI use that sample.
