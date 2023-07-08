"""
Code for reproducing Proof of Concept 2 in ACL 2023 paper:
Dealing with Semantic Underspecification in Multimodal NLP
Sandro Pezzelle
University of Amsterdam
2023

"""


import torch
import clip
from PIL import Image
import json
import os
import random
import csv
from scipy import spatial



#############

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
dir = './coco-images/images'
myseed = 42
Nsamples = 100

#############


def compute_target_random_scores():
    c=0

    F0 = 0
    F1 = 0
    F2 = 0
    F3 = 0
    F4 = 0
    F5 = 0
    F6 = 0
    F7 = 0
    F8 = 0
    F9 = 0
    F10 = 0

    f = open('100samples_with_FULL.csv', 'r')
    content = f.readlines()

    for l in content:
        cc = 0
        FgreaterR = 0
        RgreaterF = 0

        if c == 0:
           c+=1
        else:
           # print(c,l)
           c+=1
           l = l.split(';')
           
           imageURL = str(l[0])
           print(imageURL)
           imageID = str(l[1])
           full_caption = "They are doing something here."
           text_full = clip.tokenize([full_caption]).to(device)

           mypath = imageURL
           image = preprocess(Image.open(mypath)).unsqueeze(0).to(device) 

           lst100 = []
           for rnd in range(1,101):
               lst100.append(rnd)
           random.shuffle(lst100)

           for r in lst100:
               rndl = content[r]

               rndL = rndl.split(';')
               rURL = str(rndL[0])
               rID = str(rndL[1])
 
               if rURL != imageURL and rID != imageID:
                  cc += 1

                  myfields = []

                  with torch.no_grad():
                      image_features = model.encode_image(image)

                      Rcaption = str(rndL[2])                    
                      print(full_caption, Rcaption)
                      text_R = clip.tokenize([Rcaption]).to(device)

                      text_full_feats = model.encode_text(text_full)
                      text_R_feats = model.encode_text(text_R)               


                      simF = 1 - spatial.distance.cosine(image_features.cpu().numpy()[0], text_full_feats.cpu().numpy()[0])
                      simR = 1 - spatial.distance.cosine(image_features.cpu().numpy()[0], text_R_feats.cpu().numpy()[0])
                      print('full_caption:', simF)
                      print('random_caption:', simR)

                      if simF >= simR:
                         FgreaterR += 1
                      elif simR > simF:
                         RgreaterF += 1

                      myfields.append(imageURL)
                      myfields.append(imageID)
                      myfields.append(full_caption)
                      myfields.append(rURL)
                      myfields.append(rID)
                      myfields.append(Rcaption)
                      myfields.append(float(simR*2.5))




                  if cc == 10:
                     print(imageURL, 'full greater than random:', FgreaterR)
                     print(imageURL, 'random greater than full:', RgreaterF)

                     if FgreaterR == 0:
                        F0 += 1
                     elif FgreaterR == 1:
                        F1 += 1
                     elif FgreaterR == 2:
                        F2 += 1
                     elif FgreaterR == 3:
                        F3 += 1
                     elif FgreaterR == 4:
                        F4 += 1
                     elif FgreaterR == 5:
                        F5 += 1
                     elif FgreaterR == 6:
                        F6 += 1
                     elif FgreaterR == 7:
                        F7 += 1
                     elif FgreaterR == 8:
                        F8 += 1
                     elif FgreaterR == 9:
                        F9 += 1
                     elif FgreaterR == 10:
                        F10 += 1

                     break

    print('F0 R10:', F0)
    print('F1 R9:', F1)
    print('F2 R8:', F2)
    print('F3 R7:', F3)
    print('F4 R6:', F4)
    print('F5 R5:', F5)
    print('F6 R4:', F6)
    print('F7 R3:', F7)
    print('F8 R2:', F8)
    print('F9 R1:', F9)
    print('F10 R0:', F10)


if __name__ == '__main__':
    compute_target_random_scores()

