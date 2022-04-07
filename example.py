from pathlib import Path

from annotations.extract_key_clip import extract
from annotations.utils import videos_to_frame

video_dir = Path("./output_videos")
img_dir = Path("./output_imgs")


# Step 1
# draw mask and use analyze model to determine the thresbold
# mask file will be saved at ./mask.npy
# assert video_path.exists()
# extract(video_path, draw_mask=True, is_analyze=True)

# threshold need to be set to 2000


# Step 2
# extract key frames using mask file and threshold created by previous step
# save output video file under ./output directory
extract(video_path, mask_file_path="mask.npy", threshold=2000, out_dir=video_dir)


# Step 3
# extract frame from video
videos_to_frame(video_dir, img_dir)
