from PIL import Image
import imageio
import numpy as np
from threading import Thread

import os
import shutil
import sys

import time

sys.path.insert(0, "src/")

import multi_hand_tracker as mht
import plot_hand

TEMP_FOLDER = "TEMP"
N_THREADS = 8

palm_model_path = "./models/palm_detection_without_custom_op.tflite"
landmark_model_path = "./models/hand_landmark.tflite"
anchors_path = "./data/anchors.csv" 

    
def GetImageList(filename, crop=1):
    reader = imageio.get_reader(filename)
    images = []
    for index, img in enumerate(reader):
        img = Image.fromarray(img)
        w,h = img.size
        img = img.crop((w*(1-crop)/2, 0, w*(1+crop)/2, h))
        images.append(np.array(img))
        
    fps = reader.get_meta_data()['fps']
    return images, fps

def Process_Img_List(img_idx_list, thread_idx):
    
    detector = mht.MultiHandTracker(palm_model_path, landmark_model_path, anchors_path)
    
    L = len(img_idx_list)
    for c, i in enumerate(img_idx_list):
        img = img_list[i]
        kp_list, box_list = detector(img)
        ALL_KP[i] = kp_list
        ALL_BB[i] = box_list
        print ("Thread {2}: [*] Processing image {0}/{1}    \r".format(c+1, L, thread_idx), end = "")
    print ("Thread {2}: [*] Done!".format(c+1, L, thread_idx) + " "*30 + "\n", end = "")
        
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main(filename):
    
    # Initialise images
    global img_list, ALL_KP, ALL_BB
    img_list, fps = GetImageList(filename, crop=0.8)
    L = len(img_list)
    ALL_KP = [None]*L
    ALL_BB = [None]*L

    # Create TEMP folder
    if not os.path.isdir(TEMP_FOLDER): os.mkdir(TEMP_FOLDER)

    # Analyse images
    t = time.time()

    img_list_chunks = chunks(list(range(L)), L // N_THREADS)

    threads = []
    for i, chunk in enumerate(img_list_chunks):
        threads.append(Thread(target = Process_Img_List, args = (chunk, i)))
    for thread in threads: thread.start()
    for thread in threads: thread.join()

    print ("\n[+] Done!")
    print ("Time taken: {}s".format(time.time() - t))
    t = time.time()

    # Saving images
    print("\n[*] Saving images")
    count = 0
    for img, kp_list, box_list in zip(img_list, ALL_KP, ALL_BB):
        plot_hand.plot_img(img, kp_list, box_list, save=TEMP_FOLDER + r"/{}.png".format(count))
        count += 1
        print("[*] Saving image {0}/{1}    \r".format(count, L), end = "")

    print ("\n[+] Done!")
    print ("Time taken: {}s".format(time.time() - t))
    
    # Creating GIF
    print ("[*] Saving to GIF")
    images_save = []
    for i in range(L):
        images_save.append(Image.fromarray(imageio.imread(TEMP_FOLDER + r"/{}.png".format(i))))

    gif_save = images_save[0] 
    gif_save.info['duration'] = 1./fps
    gif_save.save('test2.gif', save_all=True, append_images=images_save[1:], loop=0)

    shutil.rmtree(TEMP_FOLDER)
    print ("[+] Done!")
    
if __name__ == "__main__":
    
    print ("    Input filename or url of video to process")
    print ("    E.g. https://www.signingsavvy.com/media/mp4-hd/6/6990.mp4")
    filename = input(">>>>> filename/url: ")
    main(filename)