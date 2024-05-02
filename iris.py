from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
import pickle as pk

iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42)

lin_reg = LinearRegression()
log_reg = LogisticRegression()
svc = SVC()

lin_reg = lin_reg.fit(x_train,y_train)
log_reg = log_reg.fit(x_train,y_train)
svc = svc.fit(x_train,y_train)

pk.dump(lin_reg,open('lin.pkl','wb'))
pk.dump(log_reg,open('log.pkl','wb'))
pk.dump(svc,open('svc.pkl','wb'))

