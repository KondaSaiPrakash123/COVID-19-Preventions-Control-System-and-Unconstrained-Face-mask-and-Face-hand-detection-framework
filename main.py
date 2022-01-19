import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
classes=['Face Hand Interaction','Wearing Improper Mask','Wearing Mask','Without Mask']
mypath='dataset/train/improperly_covered_face/51.jpg'
new_model = load_model('alg/Modelt.h5')
new_model.summary()
test_image = image.load_img(mypath, target_size=(224, 224))
test_image = image.img_to_array(test_image)
test_image=test_image/255
test_image = np.expand_dims(test_image, axis = 0)
result = new_model.predict(test_image)
np.argmax(result)
prediction=classes[np.argmax(result)]
print("Result is : ",prediction)
