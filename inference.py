import argparse
import torch.hub
import cv2
import os
import pandas as pd
from torchvision import transforms
from PIL import Image


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--ckpt_pth', type=str, default="")
    parser.add_argument('--labels_pth', type=str, default="")

    args = parser.parse_args()

    # load noun and verb csv
    verb_pth = os.path.join(args.labels_pth, "EPIC_100_verb_classes.csv")
    noun_pth = os.path.join(args.labels_pth, "EPIC_100_noun_classes.csv")

    verb_df = pd.read_csv(verb_pth)
    noun_df = pd.read_csv(noun_pth)

    # model 

    repo = 'epic-kitchens/action-models'

    class_counts = (125, 352)
    segment_count = 8
    base_model = 'resnet50'
    tsn = torch.hub.load(repo, 'TSN', class_counts, segment_count, 'RGB',
                        base_model=base_model, 
                        pretrained='epic-kitchens', force_reload=True)

    batch_size = 1
    segment_count = 8
    snippet_length = 1  # Number of frames composing the snippet, 1 for RGB, 5 for optical flow
    snippet_channels = 3  # Number of channels in a frame, 3 for RGB, 2 for optical flow
    height, width = 224, 224

    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
    ])

    # inputs = torch.randn(
    #     [batch_size, segment_count, snippet_length, snippet_channels, height, width]
    # )

    # inputs = inputs.reshape((batch_size, -1, height, width))

    # print(inputs.shape)
    # print(tsn(inputs))


    video_pth = "kitchen_sample.mp4"
    cap = cv2.VideoCapture(video_pth)
    count = 0
    frames_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            count += 1
            if count % 9 == 0:
                # perform inference
                frame_stack = torch.stack(frames_list, dim=0)
                inputs = frame_stack.reshape((1, -1, height, width))
                print(inputs.shape)
                # exit()

                # preprocess frame
                verb_logits, noun_logits = tsn(inputs)

                # get topk
                topK = 5
                verb_ind = verb_logits.topk(topK).indices.squeeze()
                noun_ind = noun_logits.topk(topK).indices.squeeze()

                verb = verb_df.loc[verb_df['id'] == verb_ind[0].item()].iloc[0]["key"]
                noun = noun_df.loc[noun_df['id'] == noun_ind[0].item()].iloc[0]["key"] 
                print("Action: {} {}".format(verb, noun))

                frames_list.clear()
            else:
                pil_frame = Image.fromarray(frame)
                pil_frame = transform(pil_frame)
                pil_frame = pil_frame.unsqueeze(0)
                frames_list.append(pil_frame)

        else:
            break





