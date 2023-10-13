from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from starlette.requests import Request
from aiofiles import open
from PIL import Image
import io

app = FastAPI()


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        print(file.filename)
        async with open('./buffer/saved_image.jpeg', 'wb') as buffer:
            await buffer.write(await file.read())

        file_size = await get_file_size(file.filename)
        return JSONResponse(content={"file_size": file_size}, status_code=200)

    except Exception as e:
        return HTTPException(detail=str(e), status_code=500)


@app.get("/get_image/")
async def get_image():
    try:
        async with open("./buffer/saved_image.jpeg", 'rb') as buffer:
            data = await buffer.read()
            img = Image.open(io.BytesIO(data))

            new_width = 200
            new_height = 200
            img = img.resize((new_width, new_height))

            img.save("./buffer/modified_image.jpeg")
            return FileResponse("./buffer/modified_image.jpeg", media_type="image/jpeg")

    except Exception as e:
        return HTTPException(detail=str(e), status_code=500)


async def get_file_size(file_path: str) -> int:
    try:
        async with open(file_path, 'rb') as buffer:
            data = await buffer.read()
            return len(data)
    except Exception as e:
        raise e

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
