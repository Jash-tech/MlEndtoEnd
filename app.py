from flask import Flask,render_template,jsonify,request
from src.pipelines.prediction import CustomData,PredictPrediction



app=Flask(__name__)

@app.route('/')
def homepage():
    return render_template("index.html")


@app.route('/predict',methods=["GET","POST"])
def predict_datapoint():
    if request.method=="GET":
        return render_template("form.html")
    
    else:
         data = CustomData(
            ClaimNb=int(request.form.get("ClaimNb")),
            Exposure=float(request.form.get("Exposure")),
            Power=request.form.get("Power"),
            CarAge=int(request.form.get("CarAge")),
            DriverAge=int(request.form.get("DriverAge")),
            Brand=request.form.get("Brand"),
            Gas=request.form.get("Gas"),
            Region=request.form.get("Region"),
            Density=int(request.form.get("Density")),
            ClaimFreq=float(request.form.get("ClaimFreq"))
        )
         final_data=data.get_data_as_dataframe()
         prediction=PredictPrediction()
         pred=prediction.predict(final_data)


         result=round(pred[0],4)

         return render_template("result.html",final_result=result)


if __name__=="__main__":
    app.run(host="0.0.0.0",port=8000)
