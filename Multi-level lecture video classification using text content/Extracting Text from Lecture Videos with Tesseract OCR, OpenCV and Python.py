import cv2
import pytesseract
from threading import Thread
import os.path


def video_to_text(video_name, video_content):

    video_name = video_name[:-4]       
    f = open('%s.txt' % video_name, 'a')
    f.write('\n ¿ \n')      
    f.write(video_content)
    f.close()


def video_ocr(video_name):
  
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

    vidcap = cv2.VideoCapture(video_name)
    success, image = vidcap.read()

    frame_to_sec = 30  
    count = 0              
                            
    while success:

        if count % frame_to_sec == 0:
            # image = cv2.resize(image, None, fx=0.5, fy=0.5)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 11)
            # cv2.imwrite("%s.jpg"%count, image)
            text = pytesseract.image_to_string(image, lang="eng")
            video_to_text(video_name, text)
       
        success, image = vidcap.read()
        count = count + 1


def select_video(n, m):

    for i in range(n, m):
        if os.path.exists('video%d.mp4' % n):            
            video_ocr('video%d.mp4' % n)
            
        else:
            n += 1
            

if __name__ == '__main__':

    first_thread = Thread(target=select_video, args=(1, 23))
    first_thread.start()
    
    second_thread = Thread(target=select_video, args=(23, 45))
    second_thread.start()
    
    third_thread = Thread(target=select_video, args=(45, 67))
    third_thread.start()
    
    fourth_thread = Thread(target=select_video, args=(67, 89))
    fourth_thread.start()

    fifth_thread = Thread(target=select_video, args=(89, 111))
    fifth_thread.start()
    
    print("Exiting main thread")    
