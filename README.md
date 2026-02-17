# HVI-CIDNet-cpp-onnx
The cpp and onnx version of paper: [CVPR2025 &amp;&amp; NTIRE2025] HVI: A New Color Space for Low-light Image Enhancement
(https://github.com/Fediory/HVI-CIDNet)

# python windows setting-up environment (cuda11.7+torch1.13.1)

  Package               Version<br>
--------------------- --------------------<br>
aiofiles              24.1.0<br>
annotated-doc         0.0.4<br>
annotated-types       0.7.0<br>
anyio                 4.12.1<br>
arch                  5.3.1<br>
brotli                1.2.0<br>
brotlicffi            1.2.0.0<br>
certifi               2026.1.4<br>
cffi                  2.0.0<br>
charset-normalizer    3.4.4<br>
click                 8.3.1<br>
colorama              0.4.6<br>
coloredlogs           15.0.1<br>
cycler                0.12.1<br>
einops                0.6.1<br>
exceptiongroup        1.3.1<br>
fastapi               0.129.0<br>
ffmpy                 1.0.0<br>
filelock              3.24.2<br>
flatbuffers           25.12.19<br>
fonttools             4.61.1<br>
fsspec                2026.2.0<br>
gradio                6.5.1<br>
gradio_client         2.0.3<br>
groovy                0.1.2<br>
h11                   0.16.0<br>
hf-xet                1.2.0<br>
httpcore              1.0.9<br>
httpx                 0.28.1<br>
huggingface_hub       1.4.1<br>
humanfriendly         10.0<br>
idna                  3.11<br>
image-quality         1.2.7<br>
ImageIO               2.37.2<br>
Jinja2                3.1.6<br>
kiwisolver            1.4.9<br>
libsvm                3.23.0.4<br>
lpips                 0.1.4<br>
markdown-it-py        4.0.0<br>
MarkupSafe            3.0.3<br>
matplotlib            3.5.3<br>
mdurl                 0.1.2<br>
mkl_fft               2.1.1<br>
mkl_random            1.3.0<br>
mkl-service           2.5.2<br>
ml_dtypes             0.5.4<br>
mpmath                1.3.0<br>
networkx              3.4.2<br>
np                    1.0.2<br>
numpy                 1.26.4<br>
onnx                  1.20.1<br>
onnxruntime           1.23.2<br>
onnxsim               0.4.36<br>
opencv-contrib-python 4.7.0.72<br>
opencv-python         4.11.0.86<br>
orjson                3.11.7<br>
packaging             25.0<br>
pandas                2.3.3<br>
patsy                 1.0.2<br>
Pillow                9.5.0<br>
pip                   26.0.1<br>
property-cached       1.6.4<br>
protobuf              6.33.5<br>
pycparser             2.23<br>
pydantic              2.12.5<br>
pydantic_core         2.41.5<br>
pydub                 0.25.1<br>
Pygments              2.19.2<br>
pyparsing             3.3.2<br>
pyreadline3           3.5.4<br>
PySocks               1.7.1<br>
python-dateutil       2.9.0.post0<br>
python-multipart      0.0.22<br>
pytz                  2025.2<br>
PyWavelets            1.6.0<br>
PyYAML                6.0.3<br>
requests              2.32.5<br>
rich                  14.3.2<br>
safehttpx             0.1.7<br>
safetensors           0.7.0<br>
scikit-image          0.19.3<br>
scipy                 1.7.3<br>
semantic-version      2.10.0<br>
setuptools            80.10.2<br>
shellingham           1.5.4<br>
six                   1.17.0<br>
starlette             0.52.1<br>
statsmodels           0.14.1<br>
sympy                 1.14.0<br>
thop                  0.1.1.post2209072238<br>
tifffile              2025.5.10<br>
tomlkit               0.13.3<br>
torch                 1.13.1<br>
torchaudio            0.13.1<br>
torchvision           0.14.1<br>
tqdm                  4.65.0<br>
typer                 0.23.1<br>
typer-slim            0.23.1<br>
typing_extensions     4.15.0<br>
typing-inspection     0.4.2<br>
tzdata                2025.3<br>
urllib3               2.6.3<br>
uvicorn               0.40.0<br>
wheel                 0.46.3<br>
win_inet_pton         1.1.0<br>

# Modify the atten parameter in file: net/HVI_transform.py

<img width="1350" height="940" alt="image" src="https://github.com/user-attachments/assets/9033a0a9-3c8e-4ac1-8459-fc8bd076ed01" />

the HVI_transform.py can be found in this github uploading file.

# export the model to onnx

<img width="3084" height="1206" alt="image" src="https://github.com/user-attachments/assets/49853b41-b37d-4f20-af62-0a93e9f9b5d6" />

The export_onnx.py can be found in this github uploading file.

The parameter for onnx can be setting up you. For example, the picuture width,height and opset value.

The command is: python export_onnx.py --path Fediory/HVI-CIDNet-Generalization --output HVI-CIDNet-Generalization.onnx --height 480 --width 640

PS. the --path parameter is the model,you can find all weights in https://huggingface.co/papers/2502.20272. And we choose the first item "Fediory/HVI-CIDNet-Generalization" for demostration.

<img width="3078" height="1488" alt="image" src="https://github.com/user-attachments/assets/8e377978-4419-45a5-a619-3a6b300a648d" />

# Windows onnx inference
Accoring to the main.cpp, then can run the inference sucessfully by setting the model-path, gamma value, image_path, and the final result picture savepath.<br>
The model input size must be the same as chapter "export the model to onnx" height and width values.<br>
***I sucess achieve in VS2019, cuda 12.1 and onnxrunningtime-1.18 in windows10 and windows server2012. 1.C++ inference in step 11, If you use cudnn9.x please use onnxruunming-time 18.1.1, and if you use cudnn8.x use onnx 18.1.0***
<img width="3054" height="1500" alt="image" src="https://github.com/user-attachments/assets/7cb440eb-f600-4bca-9acc-2eef84921b38" />


# Windows onnx inference results
<img width="3082" height="1586" alt="image" src="https://github.com/user-attachments/assets/c6ad68be-9e3b-4f8d-adfa-6a37e428a236" />

# Special thanks

Also Thanks for their works.<br>
    @article{yan2025hvi,
  title={HVI: A New color space for Low-light Image Enhancement},
  author={Yan, Qingsen and Feng, Yixu and Zhang, Cheng and Pang, Guansong and Shi, Kangbiao and Wu, Peng and Dong, Wei and Sun, Jinqiu and Zhang, Yanning},
  journal={arXiv preprint arXiv:2502.20272},
  year={2025}
}

@misc{feng2024hvi,
      title={You Only Need One Color Space: An Efficient Network for Low-light Image Enhancement}, 
      author={Yixu Feng and Cheng Zhang and Pei Wang and Peng Wu and Qingsen Yan and Yanning Zhang},
      year={2024},
      eprint={2402.05809},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

