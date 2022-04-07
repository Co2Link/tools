import glob
import os
import shutil
from queue import Queue

import cv2
import numpy as np
from PIL import Image, ImageDraw


class PtsGetter:
    def __init__(self, frame):
        """
        Args:
            frame: cvimage
        """
        self.frame = frame
        self.MOUST_PTS = []
        self.windowsname = "get_pts"
        self.is_finished = False

    def run(self):
        """
        Return:
            self.MOUST_PTS: list of tuple
        """
        cv2.namedWindow(self.windowsname)
        cv2.setMouseCallback(self.windowsname, self._get_mouse_points)
        while True:
            cv2.imshow(self.windowsname, self.frame)
            cv2.waitKey(1)
            if self.is_finished:
                cv2.destroyWindow(self.windowsname)
                break
        return self.MOUST_PTS

    def _get_mouse_points(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.frame, (x, y), 5, (255, 0, 0), 10)
            if len(self.MOUST_PTS) > 0:
                cv2.line(self.frame, (x, y), self.MOUST_PTS[-1], (70, 70, 70), 2)
            self.MOUST_PTS.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            cv2.line(self.frame, self.MOUST_PTS[-1], self.MOUST_PTS[0], (70, 70, 70), 2)
            pts = np.array(self.MOUST_PTS).astype("int32")
            cv2.fillPoly(self.frame, [pts], 255)
            self.is_finished = True


def extract(
    video_path,
    draw_mask=False,
    threshold=2000,
    mask_file_path=None,
    is_analyze=False,
    out_dir=None,
):
    """
    Extract key frames and write into video file and save it under out_dir
    Args:
        video_path: PosixPath or WindowsPath instantiated by pathlib.Path()
        draw_roi: draw region-of-interest, and save as mask.npy
        mask_file_path: use existing mask.npy to decide region-of-interset
        is_analyze: use analyze mode to analyze and determine the threshold, out_dir need to be set if false (press ESC to exit)
        out_dir: output video directory, PosixPath or WindowsPath instantiated by pathlib.Path()
    """
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    # if not in analyze mode, out_dir need to be set
    if not is_analyze:
        assert out_dir is not None
    assert (
        not draw_mask or mask_file_path is None
    ), "can not set mask_file_path if draw_mask is True"
    # parameters
    # the following parameters is suitable for 1 FPS video
    # pixel difference computed between T  and T + BUFFER_SIZE (T: current frame)
    # BUFFER_SIZE = 2
    # number of pixels that its pixel diference is bigger than 50, exceed which to be considered as key frame
    # THRESHOLD = threshold
    # maximum gap between frames to be considered in same section
    # MAX_GAP = 2
    # discard sections that have less than MIN_SECTION_LEN frame
    # MIN_SECTION_LEN = 3
    # add ADD_FRAME_NUM frames into the start and the end of the section
    # ADD_FRAME_NUM = 1

    # for 5 FPS video
    BUFFER_SIZE = 5
    THRESHOLD = threshold
    MAX_GAP = 3
    MIN_SECTION_LEN = 5
    ADD_FRAME_NUM = 12

    frame_idxs = []
    frame_idx = 1
    have_roi = draw_mask or mask_file_path is not None
    buf = Queue(maxsize=BUFFER_SIZE)
    cap = cv2.VideoCapture(str(video_path))
    H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ret, frame = cap.read()  # shape (800, 600)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if mask_file_path is not None:
        mask = np.load(mask_file_path)

    elif draw_mask:
        pts_getter = PtsGetter(frame_gray.copy())
        print("draw the roi")
        pts = pts_getter.run()
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).polygon(pts, outline=1, fill=1)
        mask = np.array(mask)
        with open("mask.npy", "wb") as f:
            np.save(f, mask)
    while ret:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = frame_gray.astype("int32")
        if buf.full():
            old_frame_gray = buf.get()
            buf.put(frame_gray)
            if have_roi:
                diff = np.abs(old_frame_gray - frame_gray)[mask == True]
            else:
                diff = np.abs(old_frame_gray - frame_gray).ravel()
            if is_analyze:
                frame_display = frame_gray.copy()
                # text = '50: {:<3}, 100: {:<3}, 150: {:<3}'.format(
                #     diff[diff > 50].shape[0], diff[diff > 100].shape[0], diff[diff > 150].shape[0])
                text = str(diff[diff > 50].shape[0])
                cv2.putText(
                    frame_display,
                    text,
                    (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                frame_display = cv2.addWeighted(
                    frame_display.astype("uint8"), 0.8, mask * 255, 0.2, 0
                )
                cv2.imshow("analyze", frame_display)
                k = cv2.waitKey(0)
                if k == 27:
                    break
            else:
                if diff[diff > 50].shape[0] > THRESHOLD:
                    frame_idxs.append(frame_idx)
                    print(frame_idx)
        else:
            buf.put(frame_gray)
        ret, frame = cap.read()
        frame_idx += 1
    cap.release()

    sections = []
    sect = [frame_idxs[0], None]
    previous_idx = sect[0]
    for current_idx in frame_idxs[1:]:
        if current_idx - previous_idx > MAX_GAP:
            sect[1] = previous_idx
            sections.append(sect)
            sect = [current_idx, None]
        previous_idx = current_idx

    # remove section that less that 15 frame
    sections = [sect for sect in sections if (sect[1] - sect[0]) >= MIN_SECTION_LEN]

    # add 5 frame at each section front and end
    for sect in sections:
        sect[0] -= ADD_FRAME_NUM
        sect[1] += ADD_FRAME_NUM

    write_frame_idx = []
    for sect in sections:
        write_frame_idx.extend(range(sect[0], sect[1]))

    video_name = os.path.basename(video_path)
    writer = cv2.VideoWriter(
        os.path.join(out_dir, video_name), cv2.VideoWriter_fourcc(*"mp4v"), 5, (W, H)
    )
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    counter = 1
    while ret:
        if counter in write_frame_idx:
            writer.write(frame)
        ret, frame = cap.read()
        counter += 1
    cap.release()
    writer.release()


if __name__ == "__main__":
    video_paths = glob.glob("H:/backup/rpi-1/2021-03-29_*")
    video_paths.sort()
    out_dir = "output-1"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    for path in video_paths:
        print(path)
        extract(path, mask_file_path="mask.npy", out_dir=out_dir)
