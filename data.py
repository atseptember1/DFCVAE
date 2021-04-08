import tensorflow as tf
import glob


AUTOTUNE = tf.data.experimental.AUTOTUNE 

class DataSet:
    def __init__(self, storage_dir, img_width=64, img_height=64):
        self.storage_dir = storage_dir
        self.width = img_width
        self.height = img_height
        
    def build(self, batch_size, buffer_size=10000):
        image_paths = glob.glob(self.storage_dir + "/*.jpg")
        ds = tf.data.Dataset.from_tensor_slices(image_paths)
        ds = ds.map(tf.io.read_file)
        ds = ds.map(lambda img: tf.image.decode_jpeg(img), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda img: tf.image.resize(img, [self.height, self.width]), num_parallel_calls=AUTOTUNE)
        ds = ds.cache().shuffle(buffer_size)
        ds = ds.batch(batch_size, drop_remainder=True)
        return ds, len(image_paths) // batch_size