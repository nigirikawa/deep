from keras.models import Sequential
from keras.utils import plot_model
from keras.models import load_model
from keras.models import model_from_json
import json



path = "./dir121-100-10--10000--20190628-103705/model.json"
file = open(path)
jsonStr = file.read()

model = model_from_json(jsonStr)
model.load_weights("./dir121-100-10--10000--20190628-103705/weight.txt")
plot_model(model, to_file="./dir121-100-10--10000--20190628-103705/graph.png")