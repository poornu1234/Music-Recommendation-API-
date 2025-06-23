
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from preprocess import load_and_clean
from recommender import train_model, get_recommendations

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading data + training model...")
    app.state.df = load_and_clean()
    app.state.algo = train_model(app.state.df)
    yield
    print("App shutdown.")

app = FastAPI(lifespan=lifespan)

class UserRequest(BaseModel):
    Username: str

@app.post("/recommend/")
def recommend(req: UserRequest):
    df = app.state.df
    algo = app.state.algo
    recs = get_recommendations(algo, df, req.Username)
    return {"recommendations": recs}



