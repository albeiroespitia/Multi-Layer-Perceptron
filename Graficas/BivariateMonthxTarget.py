import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

namesColumn = ["ID", "Reason for absence","Month of absence", "Day of the week", "Seasons", "Transportation expense", "Distance from Residence to Work", "Service time", "Age", "Work load Average/day", "Hit target", "Disciplinary failure", "Education", "Son", "Social drinker", "Social smoker", "Pet", "Weight","Height","Body mass index","Absenteeism time in hours"]

dataframe = pd.read_csv('./../Absenteeism_at_work.csv', names=namesColumn, sep=",",header=0)

dataframe = dataframe[dataframe["Absenteeism time in hours"] != 0]

X = dataframe.iloc[:, 0:20]
y = pd.DataFrame(dataframe["Absenteeism time in hours"])

sns.jointplot(x="Absenteeism time in hours", y="Month of absence", data=dataframe, kind="hex")
plt.show()
