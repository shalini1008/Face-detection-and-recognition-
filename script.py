import cv2

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords

def detect(img, faceCascade, EyesCascade, NoseCascade, MouthCascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    coord = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "FACE")
    if len(coord) == 4:
        roi_img = img[coord[1]: coord[1]+coord[3], coord[0]:coord[0]+coord[2]]
        draw_boundary(roi_img, EyesCascade, 1.1, 14, color['red'], "EYES")
        draw_boundary(roi_img, NoseCascade, 1.1, 5, color['green'], "NOSE")
        draw_boundary(roi_img, MouthCascade, 1.1,10, color['white'], "MOUTH")
    return img

video_capture = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
EyesCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
NoseCascade = cv2.CascadeClassifier('Nariz.xml')
MouthCascade = cv2.CascadeClassifier('Mouth.xml')

while True:
    ret, img = video_capture.read()
    img = detect(img, faceCascade, EyesCascade, NoseCascade, MouthCascade)
    cv2.imshow("Face Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()  
