import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
names = ['Area', 'Perimeter', 'Compactness', 'Length of kernel', 'Width of kernel', 'Asymmetry coefficient', 'Length of kernel groove', 'Class']
data = pd.read_csv(url, delim_whitespace=True, names=names)

data.head()

y = data['Type']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_up = scaler.fit_transform(data[['Area','Perimeter','Compactness','Kernel.Length','Kernel.Width','Asymmetry.Coeff','Kernel.Groove']])

data_up = pd.DataFrame(data_up)

data_up.head()

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(data_up,y, test_size=0.2)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train,y_train)

y_pred_log = logreg.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred_log,y_test))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred_log)
print(cm)

import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(cm,annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.xlabel('predicted labels')
plt.ylabel('actual label')
plt.show()

from sklearn.ensemble import RandomForestClassifier
randForest = RandomForestClassifier()

randForest.fit(X_train,y_train)

y_pred_ran = randForest.predict(X_test)

accuracy_score(y_test,y_pred_ran)

cm = confusion_matrix(y_test,y_pred_ran)
sns.heatmap(cm,annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.xlabel('predicted labels')
plt.ylabel('actual label')
plt.show()

from sklearn import tree
DeTree = tree.DecisionTreeClassifier()

DeTree.fit(X_train,y_train)

import graphviz

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(DeTree, out_file=dot_data,
filled=True, rounded=True,
special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

y_pred_tree = DeTree.predict(X_test)

accuracy_score(y_pred_tree,y_test)

cm = confusion_matrix(y_test,y_pred_ran)
sns.heatmap(cm,annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.xlabel('predicted labels')
plt.ylabel('actual label')
plt.show()