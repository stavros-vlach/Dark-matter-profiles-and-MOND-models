import pandas as pd

df = pd.read_csv('parameters_pISO.csv')
x = df['chi_square_emcee']
print("average chi square = ", sum(x)/len(x))
