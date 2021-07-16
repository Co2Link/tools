from annotations.extract_key_clip import extract
from pathlib import Path

video_path = Path('H:/prime42/2021-07-09_16.mp4')

# Step 1
# draw mask and use analyze model to determine the thresbold
# mask file will be saved at ./mask.npy
# assert video_path.exists()
# extract(video_path, draw_mask=True, is_analyze=True)


# Step 2
# extract key frames using mask file and threshold created by previous step
# save output video file under ./output directory
out_dir = Path('./output')
extract(video_path, mask_file_path='mask.npy', threshold=2000, out_dir=out_dir)