# Dream-Textures-Depth-Pipeline

## Installation

- Install conda and create your virtual enviroment.
- Install torch using `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`
- Install required packages by running `pip install -r requirements.txt`

## Usage

- Run python app `python app.py`
- Make a POST request to the server url or ip, fill in the parameters in the url and send the color image and the depth image by form data.

```
Form data values:
color (type File): the image color as file
depth (type File): the depth image as array - Float32[] | NDArray (eg blob image from JS)
```

```
POST http://{serverip:port | url}/depth/predict?prompt=<promtp>&strength=<float>&steps=<int>&cfg_scale=<float>&seed=<int>

Example:

POST http://216.153.52.125:8081/depth/predict?prompt=photo of a baby astronaut space walking at the international space station with earth seeing from above in the background&strength=1.0&steps=50&cfg_scale=7.5&seed=50
```

### Params

- prompt (str): The model generation based on text prompt.
- strength (float): the amount of strength the pipe will do to generate a result as similar as possible to the text prompt. Best results between 0.8 and 1.5.
- steps (int): the higher the amount the model will return a more tighter result. Results are better the more steps you use, however the more steps, the longer the generation takes. Good results between 20 and 50, best results between 75 and 100.
- cfg_scale (float): The way to increase the adherence to the conditional signal that guides the generation (text). Best results between 7 and 8.5.
- seed (int): used to generate random latent image representations. The larger the number, the more random. Good accuracy around 50.
