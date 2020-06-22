import taichi as ti
import numpy as np
from PIL import Image
import moviepy.editor as mpy

result_dir = "./results"
video_manager = ti.VideoManager(output_dir=result_dir,
                               framerate=60,
                               automatic_build=False)

maxframe = 601
for i in range(maxframe):
    imageNameExplicit = f'./ExplicitMassSpring/Frame/frame_{i:05d}.png'
    imageNameImplicit = f'./ImplicitMassSpring/Frame/frame_{i:05d}.png'
    imgEx = np.array(Image.open(imageNameExplicit))
    imgIm = np.array(Image.open(imageNameImplicit))
    pixels_img_ori = np.hstack((imgEx, imgIm))
    pixels_img = pixels_img_ori.copy()
    pixels_img = pixels_img.transpose(1,0,2)[::-1]
    pixels_img = pixels_img.reshape(int(pixels_img.size/3),3)
    pixels_img = np.array(pixels_img[::-1])
    pixels_img = pixels_img.reshape(pixels_img_ori.shape[1],
                                    pixels_img_ori.shape[0],
                                    pixels_img_ori.shape[2])
    video_manager.write_frame(pixels_img)
    print(f'\rFrame {i+1}/{maxframe} is recorded', end='')

print('\nExporting .mp4 videos...')
video_manager.make_video(gif=False, mp4=True)
print(f'MP4 video is saved to ' + video_manager.get_output_filename(".mp4").replace("\\","/"))

print('\nExporting .gif videos...')
content = mpy.VideoFileClip(video_manager.get_output_filename('.mp4').replace('\\','/'))
content = content.resize((640, 320))
content.write_gif(video_manager.get_output_filename('.gif').replace('\\','/'))
print(f'GIF video is saved to ' + video_manager.get_output_filename(".gif").replace("\\","/"))
