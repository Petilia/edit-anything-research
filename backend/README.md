# Asynchronous python backend server 


Все модели (MobileSAM, Stable Diffusion Inpainting) лежат в NVIDIA Triton Inference Server

##### Роуты:

`/upload_img/` - грузим изображение, присваиваем uuid пользователю
`/upload_points/` - получаем точки от пользователя, передаем в MobileSAM, выдаем маску, сохраняем
`/get_image/` - достаем изображения, маски и подаем в Stable Diffusion Inpainting

##### Запуск:
```
uvicorn main:app --reload --port=8012
```

![73366188617](https://github.com/Petilia/edit-anything-research/assets/79598074/2765b383-6718-4d5c-b1d7-87014d16bbd3)
