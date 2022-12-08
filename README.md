# Dream-Textures-Depth-Pipeline

## Usage

```
POST http://216.153.52.125:8081/depth/predict?prompt=<promtp>&strength=<float>&steps=<int>&cfg_scale=<float>&seed=<int>
```

### Params

prompt (str): The model generation based on text prompt.
strenght (float): the amount of strenght the pipe will do to generate a result as similar as possible to the text prompt. Best results between 0.8 and 1.5.
steps (int): the higher the amount the model will return a more tighter result. Results are better the more steps you use, however the more steps, the longer the generation takes. Good results between 20 and 50, best results between 75 and 100.
cfg_scale (float): The way to increase the adherence to the conditional signal that guides the generation (text). Best results between 7 and 8.5.
seed (int): used to generate random latent image representations. The larger the number, the more random. Good accuracy around 50.
