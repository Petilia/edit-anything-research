import nvidia.dali as dali


@dali.pipeline_def(batch_size=0, num_threads=4, device_id=0)
def pipe():
    images = dali.fn.external_source(device="cpu", name="DALI_INPUT_0")
    images = dali.fn.decoders.image(images, device="mixed")
    images = dali.fn.resize(images, resize_longer=1024)
    images = dali.fn.cast(images, dtype=dali.types.DALIDataType.FLOAT)
    return images

if __name__ == "__main__":
    pipe_instancve = pipe(batch_size=1, num_threads=4, device_id=0)
    s = pipe_instancve.serialize(filename="./model_repository/dali/1/model.dali")
