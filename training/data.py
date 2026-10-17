"""
tf.data input pipeline: TFRecord face crops → (image in [0, 1], label).

Every crop was produced by the API's own face-crop code, so a test image here
is exactly what the live model sees, minus the JPEG q95 used for storage.
"""

import tensorflow as tf

IMAGE_SIZE = 224
AUTOTUNE = tf.data.AUTOTUNE

FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
    "source": tf.io.FixedLenFeature([], tf.string),
    "group": tf.io.FixedLenFeature([], tf.string),
    "name": tf.io.FixedLenFeature([], tf.string),
    "face_px": tf.io.FixedLenFeature([], tf.int64),
}


def records(files, min_face_px: int = 0, sources=None) -> tf.data.Dataset:
    """Parsed examples with the image still JPEG-encoded, so filtering is
    cheap and only the samples actually used get decoded."""
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
    ds = ds.map(lambda r: tf.io.parse_single_example(r, FEATURES), num_parallel_calls=AUTOTUNE)
    if min_face_px:
        ds = ds.filter(lambda ex: ex["face_px"] >= min_face_px)
    if sources is not None:
        wanted = tf.constant(list(sources))
        ds = ds.filter(lambda ex: tf.reduce_any(tf.equal(ex["source"], wanted)))
    return ds


def decode(jpeg) -> tf.Tensor:
    """JPEG bytes → float32 [0, 1], the same scaling the API applies."""
    image = tf.io.decode_jpeg(jpeg, channels=3)
    image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
    return tf.cast(image, tf.float32) / 255.0


def augment(image: tf.Tensor) -> tf.Tensor:
    """Training augmentation, applied identically to real and fake crops so
    none of it can become a label cue."""
    image = tf.image.random_flip_left_right(image)

    # Face-box jitter: MTCNN boxes shift a little from frame to frame.
    size = tf.cast(tf.random.uniform([], 0.85, 1.0) * IMAGE_SIZE, tf.int32)
    image = tf.image.random_crop(image, tf.stack([size, size, 3]))
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])

    # Resolution loss: small/far faces get upscaled and look soft. Real photos
    # in DFF skew softer than the 512px renders, so randomise it for both.
    def degrade(img):
        small = tf.cast(tf.random.uniform([], 0.3, 0.9) * IMAGE_SIZE, tf.int32)
        img = tf.image.resize(img, tf.stack([small, small]), antialias=True)
        return tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])

    image = tf.cond(tf.random.uniform([]) < 0.3, lambda: degrade(image), lambda: image)

    image = tf.image.random_brightness(image, 0.08)
    image = tf.image.random_contrast(image, 0.85, 1.15)
    image = tf.image.random_saturation(image, 0.85, 1.15)
    image = tf.clip_by_value(image, 0.0, 1.0)

    # JPEG recompression: webcam frames reach the API as JPEG (q≈0.7).
    image = tf.cond(tf.random.uniform([]) < 0.7,
                    lambda: tf.image.random_jpeg_quality(image, 35, 100),
                    lambda: image)
    image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
    return image


def training_dataset(files, batch_size: int, fake_sources, min_face_px: int = 0, seed: int = 0):
    """Balanced stream: half real, half fake, fakes split evenly across the
    generators — so no generator dominates and class priors are 50/50."""
    streams = []
    for source in ["wiki"] + list(fake_sources):
        streams.append(records(files, min_face_px, [source]).shuffle(8192, seed=seed).repeat())
    weights = [0.5] + [0.5 / len(fake_sources)] * len(fake_sources)
    ds = tf.data.Dataset.sample_from_datasets(streams, weights, seed=seed)
    ds = ds.map(lambda ex: (augment(decode(ex["image"])), tf.cast(ex["label"], tf.float32)),
                num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)


def eval_dataset(files, batch_size: int, min_face_px: int = 0, sources=None, transform=None):
    """Unaugmented (image, label, source, group, name) batches, in file order."""
    def prep(ex):
        image = decode(ex["image"])
        if transform is not None:
            image = transform(image)
        return image, tf.cast(ex["label"], tf.float32), ex["source"], ex["group"], ex["name"]

    ds = records(files, min_face_px, sources).map(prep, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


# ── Test-time perturbations (applied to both classes) ─────────────────────────

def jpeg(quality: int):
    return lambda image: tf.image.adjust_jpeg_quality(image, quality)


def downscale(factor: float):
    def apply(image):
        small = int(IMAGE_SIZE * factor)
        image = tf.image.resize(image, [small, small], antialias=True)
        return tf.clip_by_value(tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE]), 0.0, 1.0)
    return apply


def gaussian_blur(sigma: float):
    radius = int(3 * sigma)
    x = tf.range(-radius, radius + 1, dtype=tf.float32)
    k = tf.exp(-(x ** 2) / (2 * sigma ** 2))
    k = k / tf.reduce_sum(k)
    kernel = (k[:, None] * k[None, :])[:, :, None, None] * tf.ones([1, 1, 3, 1])

    def apply(image):
        out = tf.nn.depthwise_conv2d(image[None], kernel, [1, 1, 1, 1], "SAME")
        return out[0]
    return apply


PERTURBATIONS = {
    "clean": None,
    "jpeg_q90": jpeg(90),
    "jpeg_q70": jpeg(70),
    "jpeg_q50": jpeg(50),
    "jpeg_q30": jpeg(30),
    "downscale_0.5": downscale(0.5),
    "downscale_0.25": downscale(0.25),
    "blur_s1": gaussian_blur(1.0),
    "blur_s2": gaussian_blur(2.0),
}
