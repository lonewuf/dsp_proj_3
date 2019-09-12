import cv2 
import numpy as np 
import os
import pymongo

count = 1
student_number = 3000

while count != 24:

  FILE_NAME = "{}.jpg".format(count)


  # Collect info from student
  student_name = input("Enter the name of the student : ")

  os.mkdir("./train_img/"+student_name)

  #insert student in database
  myclient = pymongo.MongoClient("mongodb://localhost:27017/")
  mydb = myclient["proj1db"]
  mycol = mydb["students"]

  mydict = { "name": student_name, "std_num": student_number, "scl_pts": 100 }

  img_counter = 0
  deg = -8

  try: 
      # Read image from the disk. 
      img = cv2.imread(FILE_NAME) 
    

      while img_counter != 31:
        # Shape of image in terms of pixels. 
        (rows, cols) = img.shape[:2] 
      
        # getRotationMatrix2D creates a matrix needed for transformation. 
        # We want matrix for rotation w.r.t center to 45 degree without scaling. 
        img_name = "ActiOn_{}.jpg".format(img_counter)

        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), deg, 1) 
        res = cv2.warpAffine(img, M, (cols, rows)) 
      
      
        # Write image back to disk. 
        cv2.imwrite("train_img/"+student_name+"/"+img_name, res)
        img_counter += 1
        deg += .5
  except IOError: 
      print ('Error while reading files !!!') 

  print(student_name + " is inserted on database")

  count += 1
  student_number += 1

  x = mycol.insert_one(mydict)
  print(x.inserted_id)
  

print("Finish")
