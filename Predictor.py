import pandas
import array
from sklearn.linear_model import LinearRegression
print("Note:Database file must be located into Folder of Program")
filename=input('Enter the filename.csv formate:')
db=pandas.read_csv(filename)
db.info()
db.columns
col_len=len(db.columns)
i=int(input("Enter the number of column which we use as Feature:"))
x_tag=db.columns[i]
x=db[db.columns[i]].values.reshape(-1,1)
j=int(input("Enter the number of column which we use as Independent Varible:"))
y_tag=db.columns[j]
y=db[db.columns[j]]
mind=LinearRegression()
mind.fit(x,y)
k=int(input('Enter the {} for Predict a {}:'.format(x_tag,y_tag)))
print('Predicted value')
print(mind.predict([[k]]))
print("Coefficient or Weight :")
print(mind.coef_)