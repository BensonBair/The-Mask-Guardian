from keras import Sequential, models
from scipy.spatial import distance
import numpy as np
import cv2
import matplotlib.pyplot as plt
import schedule
import time
from os import listdir, path
import time
import shutil

INPUT_PATH = 'C:\\Users\\Asus\\Desktop\\OOP_FinalProject\\input'
OUTPUT_PATH = 'C:\\Users\\Asus\\Desktop\\OOP_FinalProject\\output'
PIC_EXT = ["jpg", "jpeg", "png"]

def get_finetuned_model():
    print("loading model...")
    model = models.load_model(f"{INPUT_PATH}\\finetuned_model\\masknet.h5")
    print("model loaded.")
    if model:
        return model
    return None

def get_face_model():
        face_model = cv2.CascadeClassifier(f"{INPUT_PATH}\\Haarcascades\\haarcascade_frontalface_default.xml")
        if face_model:
            return face_model
        return None
    
class detector():

    def __init__(self, model, face_model, filename):
        self.model = model
        self.face_model = face_model
        self.filename, self.ext = filename.split(".")
        self.timestamp = time.strftime("%Y%m%d%H%M%S")
        self.img = cv2.imread(f"{INPUT_PATH}\\CapturedPhoto\\{self.filename}.{self.ext}")
        self.out_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
        self.faces = self._get_faces()
    
    def _greyscale_img(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    # parameters matter!
    def _get_faces(self):
        return self.face_model.detectMultiScale(self._greyscale_img(), scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    def plot_faces(self):
        for (x, y, w, h) in self.faces:
            cv2.rectangle(self.out_img, (x, y), (x+w, y+h), (127, 255, 0), 3)
        plt.figure(figsize=(12, 12))
        plt.imshow(self.out_img)
        plt.show()

    def save_img_with_text(self):
        mask_label = {0: 'Mask', 1: 'No Mask!'}
        dist_label = {0: (0, 255, 0), 1: (255, 0, 0)}
        without_mask = 0
        label = [0 for i in range(len(self.faces))]

        for i in range(len(self.faces)):
            (x, y, w, h) = self.faces[i]
            crop = self.out_img[y:y+h, x:x+w]
            crop = cv2.resize(crop, (128, 128))
            crop = np.reshape(crop, [1, 128, 128, 3])/255.0
            mask_prob = self.model.predict(crop)
            predict_label = mask_prob.argmax()
            cv2.putText(self.out_img, mask_label[predict_label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, dist_label[label[i]], 2)
            cv2.rectangle(self.out_img, (x, y), (x+w, y+h), dist_label[label[i]], 1)
            if predict_label:
                without_mask = 1
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(self.out_img)
        # plt.show()
        plt.draw()
        if without_mask:
            fig.savefig(f"{OUTPUT_PATH}\\WithoutMask\\{self.filename}_{self.timestamp}.{self.ext}")
        else:
            fig.savefig(f"{OUTPUT_PATH}\\WithMask\\{self.filename}_{self.timestamp}.{self.ext}")

def get_target_pics(file_path):
    pic_tuple = tuple(PIC_EXT)
    new_pics = [f for f in listdir(file_path) if (path.isfile(path.join(file_path, f))) and f.endswith(pic_tuple)]
    return new_pics

        
def update_routine(model, face_model):
    new_pics = get_target_pics(f"{INPUT_PATH}\\CapturedPhoto")
    for pic_fname in new_pics:
        print(f"processing: {pic_fname}")
        obj_detector = detector(model, face_model, pic_fname)
        obj_detector.save_img_with_text()
        timestamp = time.strftime("%Y%m%d%H%M%S")
        new_pic_fname = f"_{timestamp}.".join(pic_fname.split("."))
        shutil.move(f"{INPUT_PATH}\\CapturedPhoto\\{pic_fname}", f"{INPUT_PATH}\\DonePhoto\\{new_pic_fname}") 

def run_main():
    print("run main")
    
    model = get_finetuned_model()
    face_model = get_face_model()
    
    schedule.every(3).seconds.do(update_routine,  model = model, face_model = face_model)
    print("schehule done.")
    while True:
        schedule.run_pending()
        time.sleep(1)
        print("sleeping...")

if __name__ == '__main__':
    run_main()


    
