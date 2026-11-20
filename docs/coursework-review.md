# Coursework review

**Module:** Predictive Analytics and Machine Learning using Python, M.Sc. Data Analytics, BSBI, Term 1 (submitted June 2024)
**Original:** one Jupyter notebook (Google Colab) and a written report

## What the original did

* Loaded the airline survey after some cleaning in Excel. The notebook shows 23,616 rows.
* Filled missing values with the median or mode, drew box plots, removed outliers by IQR for the charts only, and drew a correlation heat map.
* Trained one random forest inside a scikit-learn pipeline (scaling and one-hot encoding) on a 70/30 split.
* Reported 99% accuracy and a confusion matrix, and recommended the random forest.

## What I found when I rebuilt it

| Issue | Effect | Fix in this repository |
|---|---|---|
| Only the first 23,616 of 129,880 rows were used | Those rows hold more 0 ("not rated") answers, and a 0 on seat comfort almost always means satisfied. They are easier: the same model scores 98.9% on them, 94.3% on a random 23,616 rows and 95.5% on all rows. | Full data set, shuffled stratified split, and a row-order check in the report |
| One model, no comparison | The report says the random forest was chosen over other models, but no other model was trained | Logistic regression, random forest and gradient boosting compared on a validation set |
| No separate validation set | Model choice and final score came from the same test set | 60/20/20 split; the test set is used once |
| Report says an 80/20 split, code uses 70/30 | Mismatch between report and code | One split, set in code and stated in the results |
| Missing values filled before the split | Small leak of test information into training | Imputation inside each model pipeline |
| Data source given as the US BTS Form 41 data | The file is the Kaggle *Airlines Customer satisfaction* survey, which is not BTS data | Source stated correctly in `data/README.md` |
| The business question (which services to fix) was not answered with numbers | The report ended with model accuracy only | Permutation importance, rating gaps, and a short recommendation |
| Rating 0 treated as a score | 0 means "not rated" and behaves very differently from 1 | Documented, shown in a table, and left out of the rating-gap analysis |

## What I kept

The question from the original report: which part of the passenger experience is most linked to satisfaction, so an airline can focus its effort, in the spirit of the 80/20 rule. The cumulative-share column in the driver table answers it directly: seat comfort, customer type and inflight entertainment cover about 80% of the model's importance.

## Numbers side by side

| | Original | Rebuilt |
|---|---|---|
| Rows used | 23,616 | 129,880 |
| Models compared | 1 | 3 |
| Reported accuracy | 99% (random forest) | 95.5% (gradient boosting), 95.3% (random forest) |
| Validation set | No | Yes |
| Drivers quantified | No | Yes |
