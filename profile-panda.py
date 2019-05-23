{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import pandas_profiling\n",
    "namesColumn = [\"ID\", \"Reason for absence\",\"Month of absence\", \"Day of the week\", \"Seasons\", \"Transportation expense\", \"Distance from Residence to Work\", \"Service time\", \"Age\", \"Work load Average/day \", \"Hit target\", \"Disciplinary failure\", \"Education\", \"Son\", \"Social drinker\", \"Social smoker\", \"Pet\", \"Weight\",\"Height\",\"Body mass index\",\"Absenteeism time in hours\"]\n",
    "\n",
    "\n",
    "dataframe = pd.read_csv('Absenteeism_at_work.csv', names=namesColumn, sep=\",\",header=0)\n",
    "report = pandas_profiling.ProfileReport(dataframe)\n",
    "report.to_file(\"report.html\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
