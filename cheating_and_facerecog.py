from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
import pymongo
import dlib
import numpy as np 
import datetime
from imutils import face_utils

## Face recog variables
modeldir = './model/20170511-185253.pb'
classifier_filename = './class/classifier.pkl'
npy='./npy'
train_img="./train_img"


saved_name = "sample name"


myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["proj1db"]
mycol = mydb["students"]


## Head pose variables
face_landmark_path = './shape_predictor_68_face_landmarks.dat'

K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


def countdown(t):
    while t:
        mins, secs = divmod(t, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)
        print(timeformat, end='\r')
        time.sleep(1)
        t -= 1

def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle



with tf.Graph().as_default():
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160

        
        HumanNames = os.listdir(train_img)
        HumanNames.sort()

        print('Loading Model')
        facenet.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        print(embeddings)
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]


        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile, encoding='latin1')

        video_capture = cv2.VideoCapture(0)
        c = 0

        ## custom
        num_detected = 0
        name_detected = "Sample name"

        print('Start Recognition')
        prevTime = 0
        checker = True
        while checker:
            ret, frame = video_capture.read()

            frame = cv2.resize(frame, (0,0), fx=0.8, fy=0.8)    #resize frame (optional)

            curTime = time.time()+1    # calc fps
            timeF = frame_interval

            if (c % timeF == 0):
                find_results = []

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                print('Detected_FaceNum: %d' % nrof_faces)

                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces,4), dtype=np.int32)

                    for i in range(nrof_faces):
                        emb_array = np.zeros((1, embedding_size))

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('Face is very close!')
                            continue

                        cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        cropped[i] = facenet.flip(cropped[i], False)
                        scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                        scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                               interpolation=cv2.INTER_CUBIC)
                        scaled[i] = facenet.prewhiten(scaled[i])
                        scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                        feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                        predictions = model.predict_proba(emb_array)
                        # print(predictions)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        # print("predictions")
                        # print(best_class_indices,' with accuracy ',best_class_probabilities)

                        print(best_class_probabilities)
                        if best_class_probabilities>0.30:
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face
                            
                            #plot result idx under box
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20
                            
                            print('Result Indices: ', best_class_indices[0])
                            print(HumanNames)
                            print(HumanNames[best_class_indices[0]])

                            ## custom
                            if num_detected <= 5:
                                if name_detected == HumanNames[best_class_indices[0]]:
                                    num_detected+=1
                                else:
                                    name_detected = HumanNames[best_class_indices[0]]
                                    num_detected = 0
                            else:
                                saved_name = HumanNames[best_class_indices[0]]
                                num_detected = 0
                                checker = False
  

                            for H_i in HumanNames:
                                if HumanNames[best_class_indices[0]] == H_i:
                                    # result_names = HumanNames[best_class_indices[0]]
                                    resultN = "{}: {}".format(HumanNames[best_class_indices[0]],best_class_probabilities)
                                    cv2.putText(frame, resultN, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                .7, (0, 0, 255), thickness=1, lineType=2)
                        else:
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face
                            #plot result idx under box
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20
                            cv2.putText(frame, "Unknown", (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 0, 0), thickness=1, lineType=2)
                else:
                    print('Alignment Failure')
            # c+=1
            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Unable to connect to camera.")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(face_landmark_path)
dur_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        face_rects = detector(frame, 0)

        if len(face_rects) > 0:
            shape = predictor(frame, face_rects[0])
            shape = face_utils.shape_to_np(shape)

            reprojectdst, euler_angle = get_head_pose(shape)

            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            for start, end in line_pairs:
                cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))

            cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 0), thickness=2)
            cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 0), thickness=2)
            cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 0), thickness=2)

            print("Y: " + "{:7.2f}".format(euler_angle[1, 0]))
            current_angle = float("{:7.2f}".format(euler_angle[1, 0]))

            if current_angle >= 20 or current_angle <= -20:
                print("cheating")
                dur_count += 1
                print(dur_count)
                if dur_count >= 8:
                    print(saved_name + " caught cheating!")
                    dur_count = 0
                    myquery = { "name": saved_name }
                    newvalues = { "$inc": { "scl_pts": -2 }}
                    x = mycol.update_one(myquery, newvalues)
                    mycol = mydb["logs"]
                    myquery2 = { "name": saved_name, "scl_pts": -2, "violation": "Head turn", "date": datetime.datetime.now() }
                    y = mycol.insert_one(myquery2)
                    print(saved_name+"'s social points is deducted")
                    countdown(5)
            else:
                dur_count = 0

        cv2.imshow("Cheating Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
