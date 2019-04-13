import pandas as pd

def normalize(x):
    try:
        x = x/np.linalg.norm(x,ord=1)
        return x
    except :
        raise

#   The naome of the dataset downloaded from Kaggle
credit_data = pd.read_csv('creditcard.csv')

norm_cols = []
for i in range(1, 29):
    norm_cols.append('V' + str(i))

normalized_data = pd.DataFrame.apply(credit_data[norm_cols], normalize)

time = credit_data.Time
amount = credit_data.Amount
y = credit_data.Class

normalized_data['Time'] = time
normalized_data['Amount'] = amount
normalized_data['Class'] = y

#   Storing the normalized data in a new csv.
normalized_data.to_csv('creditcard_normalized.csv', index=False)
