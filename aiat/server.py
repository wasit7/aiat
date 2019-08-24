from aiohttp import web
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import numpy as np
loaded_model = pickle.load(open("titanic.model", 'rb'))

async def handle(request):
    x = request.match_info.get('x', '0,0,0,0,0,0,0')
    x_list=x.split(",")
    x_vector=np.zeros(7)
    for i in range(7):
        x_vector[i]=float(x_list[i])
    y=loaded_model.predict([x_vector])
    return web.Response(text=str(y))

app = web.Application()
app.add_routes([web.get('/predict/{x}', handle)])

web.run_app(app,port=8081)