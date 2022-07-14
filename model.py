import cv2
import numpy as np
import pandas as pd
import face_recognition
import matplotlib.pyplot as plt
import os
import datetime
#Improting deepface library
from deepface import DeepFace


def model():
    # In[2]:
    # preparing Auto detection
    path = 'ImgAtdc'
    images = []
    classNames = []
    myList = os.listdir(path)
    #print(myList)
    
    
    # In[3]:
    
    
    # code to read images and store it in array 
    # and code to remove jpg extensions from name
    for i in myList:
        curImg = cv2.imread(path + '\\'+ i)
        images.append(curImg)
        classNames.append(os.path.splitext(i)[0])
    #print(classNames)
    
    
    # In[4]:
    
    
    # function that performes encoding
    def findEncoding(images):
        encodedList = []
        for i in images:
            img = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
            encodeFace = face_recognition.face_encodings(img)[0]
            encodedList.append(encodeFace)
        return(encodedList)
    
    
    # In[5]:
    
    
    # trying out encoding function
    encodedList = findEncoding(images)
    #print(len(encodedList))
    
    
    # ### how the code will work
    # 1. On pressing button 'Start button' on UI the webcam will start and show the webcam video
    # 2. Press 'c' to capture the image and timestamp of the image whose record will be made in csv
    # 3. if 'c' is pressed multiple times, multiple records will be made and added to csv
    # 4. On pressing q the video will stop capturing and the csv is now availbale to downlaod
    
    # In[26]:
    
    
    df = []
    #Starting the webcam
    cap = cv2.VideoCapture(1)
    #Checking status of webcam and throw error is webcam not able to activate
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened:
        raise IOError("Cannot Start Webcam")
    
    while True:
        ret,frame = cap.read()
        #printing sentiment
        #writing the emotion in the image using deepface
        predictions = DeepFace.analyze(frame,enforce_detection = False,actions = ['emotion'])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame , predictions['dominant_emotion'], (0,25),font,1, (0, 0, 255),2,cv2.LINE_4);
        #inserting timestamp
        d = datetime.datetime.now()
        cv2.putText(frame , str(d), ((frame.shape[1]-400),(frame.shape[0]-10)),font,1, (0, 255, 255),2,cv2.LINE_4);
        
        
        #reducing image size for faster processing
        imgS = cv2.resize(frame,(0,0),None,0.25,0.25)
        imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
        
        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)
        
        # looping the current frame in video through database image
        
        for encodeFace,faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodedList,encodeFace, tolerance=0.6)
            #face_recognition.api.compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6)
            #Lower is stricter. 0.6 is showing best results here
            faceDis = face_recognition.face_distance(encodedList,encodeFace)
            #print(matches,'\t',faceDis)
            # getting the index with the lowest faceDis value
            #Low FaceDis value means highly matched images
            matchIndex = np.argmin(faceDis)
            
            #getting the names
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                #print(name)
                y1,x2,y2,x1 = faceLoc
                #Multiplying by 4 because initially we reducce size to 25% for faster process
                y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                #draw rectangle on image
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                #creating rectangle to write names
                cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(frame,name,(x1+6,y2-6),font,1,(255,255,255),3)
                
                
        cv2.imshow('WebCam',frame)    
        key = cv2.waitKey(1)
        if key == ord('q'):
            #plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            break
        elif key == ord('c'):
            df.append([name,predictions['dominant_emotion'],d])
            plt.figure()
            plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        
    cap.release()
    cv2.destroyAllWindows()
    
    
    # In[28]:
    
    
    df1 = pd.DataFrame(df)
    if not df1.empty:
        df1.columns = ['Name','Sentiment','Timestamp']
        df1.index += 1
    else:
        df1 = pd.DataFrame(['Please capture some data to create records'])
        df1.columns = ['Error']
    record = df1
    return(record)