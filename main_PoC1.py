"""
Code for reproducing Proof of Concept 1 in ACL 2023 paper:
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
dir = './coco-images' # contains two subfolders: images and annotations
myseed = 42
Nsamples = 100


#############



def select_samples(directory, N_samples, seed):
    myIDs = []

    out_txt = open('orig_captions_100_images.txt', 'w')

    for path in os.listdir(dir):
        # print(path) # actual image urls
        idfull = str(path).split('.')[0].split('_')[-1]
        idshort = str(idfull)[6:]
        if str(idshort).startswith('0'):
           idshort = str(idshort)[1:]
        myIDs.append(str(idshort))
    # print(myIDs) # this is a list with all image IDs

    random.seed(myseed)
    random.shuffle(myIDs)
    rand100 = myIDs[:Nsamples-1]

    # print(rand100)

    f = open('./coco-images/annotations/captions_train2014.json')
    data = json.load(f)
    my100 = []
    with open('json100.json', 'w') as outfile:
        for i in data['annotations']:
            if str(i['image_id']) in rand100:
               print(i)
               my100.append(i)
               # json.dump(i, outfile)
               out_txt.write(str(i)+'\n')

        json.dump(my100, outfile)
        print(len(my100))
    return outfile, out_txt


def compute_scores():
    f = open('json100.json', 'r')
    data = json.load(f)
    myoutput = open('clip_scores_100_samples.csv', 'w')
    writer = csv.writer(myoutput)
    
    for i in data:
        print(str(i['image_id']))
        # print(len(str(i['image_id'])))
        print(str(i['caption']))

        target_caption = str(i['caption'])
        if len(str(i['image_id'])) == 5:
           longID = 'COCO_train2014_0000000'+str(i['image_id'])+'.jpg'
        elif len(str(i['image_id'])) == 6:
           longID = 'COCO_train2014_000000'+str(i['image_id'])+'.jpg'

        mypath = './coco-images/images/'+str(longID)
        image = preprocess(Image.open(mypath)).unsqueeze(0).to(device)
        text = clip.tokenize([target_caption]).to(device)

        # print(text)

        myfields = []

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            # print(image_features.cpu().numpy())
            sim = 1 - spatial.distance.cosine(image_features.cpu().numpy()[0], text_features.cpu().numpy()[0])

            sim1, sim2 = model(image, text)

            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            print("Sim:", sim)
            print("Sim2:", sim1, sim2)
            print('CLIPScore:', sim*2.5)
            # print("Label probs:", probs)

            myfields.append(mypath)
            myfields.append(str(i['image_id']))
            myfields.append(target_caption)
            myfields.append(str(sim))
            myfields.append(str(sim1[0]))
            myfields.append(str(sim*2.5))

            # writer = csv.writer(file)
            writer.writerow(myfields)

    return sim, sim1, sim2


def compute_und_scores():
    c=0
    f = open('100samples_with_FULL.csv', 'r')
    content = f.readlines()

    myoutput = open('clip_UND_scores_100_samples.csv', 'w')
    myoutput.write('imageURL,imageID,original,und_quantity,und_location,und_object,und_gender-number,und_gender,und_full,sim_original,sim_quantity,sim_location,sim_object,sim_gender-number,sim_gender,sim_full\n')
    writer = csv.writer(myoutput)

    for l in content:
        if c == 0:
           c+=1
        else:
           print(c,l)
           c+=1
           l = l.split(';')
           
           imageURL = str(l[0])
           print(imageURL)
           imageID = str(l[1])
           orig_caption = str(l[2])
           und_some = str(l[3])
           und_locative = str(l[4])
           und_demonstrative = str(l[5]) 
           und_they = str(l[6])
           und_person = str(l[7]) 
           und_full = str(l[8]) 

           mypath = imageURL
           image = preprocess(Image.open(mypath)).unsqueeze(0).to(device)

           text_O = clip.tokenize([orig_caption]).to(device)
           text_some = clip.tokenize([und_some]).to(device)
           text_loc = clip.tokenize([und_locative]).to(device)
           text_dem = clip.tokenize([und_demonstrative]).to(device)
           text_they = clip.tokenize([und_they]).to(device)
           text_person = clip.tokenize([und_person]).to(device)
           text_full = clip.tokenize([und_full]).to(device)
           
           myfields = []

           with torch.no_grad():
               image_features = model.encode_image(image)

               text_O_feats = model.encode_text(text_O)
               
               sim0 = 1 - spatial.distance.cosine(image_features.cpu().numpy()[0], text_O_feats.cpu().numpy()[0])
               print(orig_caption)
               # print('original_caption:', sim0)

               if und_some != 'NA':
                  text_some_feats = model.encode_text(text_some)
                  sim1 = 1 - spatial.distance.cosine(image_features.cpu().numpy()[0], text_some_feats.cpu().numpy()[0])
                  # print(und_some)
                  # print('some_caption:', sim1)
               else:
                  sim1 = 0

               if und_locative != 'NA':
                  text_loc_feats = model.encode_text(text_loc)
                  sim2 = 1 - spatial.distance.cosine(image_features.cpu().numpy()[0], text_loc_feats.cpu().numpy()[0])
                  # print(und_locative)
                  # print('locative_caption:', sim2)
               else:
                  sim2 = 0


               if und_demonstrative != 'NA':
                  text_dem_feats = model.encode_text(text_dem)
                  sim3 = 1 - spatial.distance.cosine(image_features.cpu().numpy()[0], text_dem_feats.cpu().numpy()[0])
                  # print(und_demonstrative)
                  # print('demonstrative_caption:', sim3)
               else:
                  sim3 = 0


               if und_they != 'NA':
                  text_they_feats = model.encode_text(text_they)
                  sim4 = 1 - spatial.distance.cosine(image_features.cpu().numpy()[0], text_they_feats.cpu().numpy()[0])
                  # print(und_they)
                  # print('they_caption:', sim4)
               else:
                  sim4 = 0
               
               if und_person != 'NA':
                  text_person_feats = model.encode_text(text_person)
                  sim5 = 1 - spatial.distance.cosine(image_features.cpu().numpy()[0], text_person_feats.cpu().numpy()[0])
                  # print(und_person)
                  # print('person_caption:', sim5)
               else:
                  sim5 = 0

               if und_full != 'NA':
                  text_full_feats = model.encode_text(text_full)
                  sim6 = 1 - spatial.distance.cosine(image_features.cpu().numpy()[0], text_full_feats.cpu().numpy()[0])
                  # print(und_full)
                  # print('full_caption:', sim6)
               else:
                  sim6 = 0

               myfields.append(imageURL)
               myfields.append(imageID)
               myfields.append(orig_caption)
               myfields.append(und_some)
               myfields.append(und_locative)
               myfields.append(und_demonstrative)
               myfields.append(und_they)
               myfields.append(und_person)
               myfields.append(und_full)
               myfields.append(float(sim0*2.5))
               myfields.append(float(sim1*2.5))
               myfields.append(float(sim2*2.5))
               myfields.append(float(sim3*2.5))
               myfields.append(float(sim4*2.5))
               myfields.append(float(sim5*2.5))
               myfields.append(float(sim6*2.5))


               writer.writerow(myfields)
               print('written line:', c)


if __name__ == '__main__':

    # to obtain results of PoC1 reported in the paper, Figure 2
    compute_und_scores() 



