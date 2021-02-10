from flask import *
import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

app=Flask(__name__)

@app.route("/")
@app.route("/home",methods=["GET","POST"])
def home():
	if request.method=="POST":
		sl=float(request.form['SL'])
		sw=float(request.form['SW'])
		pl=float(request.form['PL'])
		pw=float(request.form['PW'])
		path="iris123.csv"
		names=['sl','sw','pl','pw','class']

		data=pandas.read_csv(path,names=names)
		X=data.iloc[:,:4].values
		y=data.iloc[:,4].values

		x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.33)

		model=KNeighborsClassifier(n_neighbors=3)
		model.fit(x_train,y_train)
		y_pred=model.predict(x_test)
		#print(y_pred)
		test=model.predict([[sl,sw,pl,pw]])
		acc=accuracy_score(y_test,y_pred)*100
		a=test[0]
		out="It may be " + test[0] + " with an accuracy of "+str(acc)+" % "
		return render_template("home.html",acc=acc,test=out,a=a)



	return render_template("home.html")





if __name__=="__main__":
	app.run(debug=True)