# color-to-depth

Repository of bouding box coordinates mapping from color image to depth frame. It was created for the question on Intel community (https://support.intelrealsense.com/hc/en-us/community/posts/360033564014)
There is a problem that the result of projection are not correct.

## Dependencies
opencv 3.4.1, pytorch (v1.0.0), numpy (1.16.1), pyrealsense2

## Model 
This repository use pretrained mtcnn face detector from https://github.com/TropComplique/mtcnn-pytorch

## Usage
```
git clone https://github.com/AndreyMaslow/color-to-depth

cd color-to-depth

python3 example.py
```
