import pandas as pd

df = pd.read_csv('/home/stavros/Documents/Pythonproject/parameters_pISO.csv')
x = df['chi_square_emcee']
print("average chi square = ", sum(x)/len(x))
