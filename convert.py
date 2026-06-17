import tensorflow as tf
model = tf.keras.models.load_model("models/best_dl_model.keras")
model.save("models/best_dl_model.h5")
print("Done! Size check models folder now")