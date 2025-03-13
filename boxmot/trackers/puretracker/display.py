import os
import multiprocessing as mp

import numpy as np
import cv2
from matplotlib import pyplot as plt

from boxmot.trackers.puretracker.utils import xywh2xyxy_clip

def mouse_callback(event, x, y, flags, param):
    """
    Handle mouse events for the visualization window

    Args:
        event: Mouse event type
        x, y: Mouse coordinates
        flags: Event flags
        param: Additional parameters (cell positions, crops, etc.)
    """
    if event == cv2.EVENT_MOUSEMOVE:
        vis_img, cell_positions, track_crops, detection_crops, popup_name = param

        # Adjust y coordinate to account for title bar
        y = y - 32  # Adjusted for title bar height

        # Check if mouse is over a cell
        for cell in cell_positions:
            if (cell['x1'] <= x < cell['x2'] and
                    cell['y1'] <= y < cell['y2']):

                # Get track and detection indices
                track_id = cell['track_id']
                track_idx = cell['track_idx']
                det_idx = cell['det_idx']

                # Check if indices are valid
                if track_idx < len(track_crops) and det_idx < len(detection_crops):
                    # Get crops
                    track_crop = track_crops[track_idx]
                    det_crop = detection_crops[det_idx]

                    # Create popup window with both images side by side
                    popup_height = max(track_crop.shape[0], det_crop.shape[0])
                    # 10px spacing
                    popup_width = track_crop.shape[1] + det_crop.shape[1] + 10
                    popup = np.full(
                        (popup_height, popup_width, 3), 240, dtype='uint8')

                    # Add track crop
                    popup[:track_crop.shape[0],
                          :track_crop.shape[1]] = track_crop

                    # Add detection crop
                    popup[:det_crop.shape[0], track_crop.shape[1]+10:] = det_crop

                    # Add labels
                    cv2.putText(popup, f"Track {track_id}", (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(popup, f"Detection {det_idx}", (track_crop.shape[1]+20, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Add distance metrics
                    metrics_y = popup_height - 20
                    cv2.putText(popup, f"IoU: {cell['iou']}", (10, metrics_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(popup, f"Emb: {cell['emb']}", (track_crop.shape[1]+20, metrics_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128), 2)

                    cv2.imshow(popup_name, popup)
                    cv2.resizeWindow(popup_name, popup_width, popup_height)
                    
                    return

        # If mouse is not over any cell, close popup window
        try:
            cv2.destroyWindow(popup_name)
        except:
            pass  # Window may not exist


# --- GUI Manager ---
class GUIManager:
    def __init__(self):
        self.queue = mp.Queue()
        self.process = mp.Process(target=self._gui_loop, args=(self.queue,), daemon=True)
        self.process.start()

    def _gui_loop(self, queue):
        """Handles OpenCV window updates in a separate process."""
        callbacks = {}  # Store callbacks for windows
        active_windows = {}  # Map window_name -> window_size

        while True:
            while not queue.empty():
                msg = queue.get()
                
                if msg[0] == 'update':
                    window_name, frame, window_size = msg[1], msg[2], msg[3]
                    
                    # If the window isn’t created yet, create it with WINDOW_NORMAL
                    if window_name not in active_windows:
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        active_windows[window_name] = window_size
                        # Immediately set the window size
                        if window_size is not None:
                            cv2.resizeWindow(window_name, window_size[0], window_size[1])
                    else:
                        # If window_size changed, update it
                        if window_size is not None and active_windows[window_name] != window_size:
                            cv2.resizeWindow(window_name, window_size[0], window_size[1])
                            active_windows[window_name] = window_size

                    # Show the full-resolution image (it will not be automatically scaled)
                    cv2.imshow(window_name, frame)
                    
                    # Attach mouse callback if registered
                    if window_name in callbacks:
                        cb, param = callbacks[window_name]
                        cv2.setMouseCallback(window_name, cb, param)
                    
                elif msg[0] == 'set_callback':
                    window_name, callback, param = msg[1], msg[2], msg[3]
                    callbacks[window_name] = (callback, param)
                    if window_name in active_windows:
                        cv2.setMouseCallback(window_name, callback, param)
                
                elif msg[0] == 'destroy':
                    window_name = msg[1]
                    cv2.destroyWindow(window_name)
                    active_windows.pop(window_name, None)
                    callbacks.pop(window_name, None)
                
                elif msg[0] == 'quit':
                    cv2.destroyAllWindows()
                    return

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def show(self, window_name, frame, window_size=None):
        """Send a frame to the GUI process for display."""
        self.queue.put(('update', window_name, frame, window_size))

    def set_callback(self, window_name, callback, param):
        """Register a mouse callback, ensuring the window exists first."""
        self.queue.put(('set_callback', window_name, callback, param))

    def destroy(self, window_name):
        self.queue.put(('destroy', window_name))

    def quit(self):
        self.queue.put(('quit',))
        self.process.join()


gui_manager = GUIManager()


def get_object_crop(img, box, crop_size=(128, 256)):
    """
    Extract and resize an object crop from an image based on bounding box.

    Args:
        img (numpy.ndarray): Image to crop from
        box (list): Bounding box in format [x1, y1, x2, y2]
        crop_size (tuple): Size of the crop (width, height)

    Returns:
        numpy.ndarray: Cropped and resized image
    """
    x1, y1, x2, y2 = [int(coord) for coord in box]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1] - 1, x2)
    y2 = min(img.shape[0] - 1, y2)

    if x2 > x1 and y2 > y1:
        crop = img[y1:y2, x1:x2]
        crop = cv2.resize(crop, crop_size)
        return crop
    else:
        # Return blank image if bbox is invalid
        return np.full((crop_size[1], crop_size[0], 3), 225, dtype='uint8')


def visualize_tracking(img, t_boxes, d_boxes, dets_confs, iou_dists, emb_dists = None, 
                       title="Tracking Visualization", crop_size=(128, 256), t_ids=None):
    """
    Visualize tracking results with distance metrics between tracks and detections.
    Now, detections are on columns (top) and tracks are on rows (left).

    Args:
        prev_img (numpy.ndarray): Previous frame image
        t_boxes (list): List of bounding boxes for tracks in format [x1, y1, x2, y2]
        cur_img (numpy.ndarray): Current frame image
        d_boxes (list): List of bounding boxes for detections in format [x1, y1, x2, y2]
        iou_dists (numpy.ndarray): IoU distance matrix between tracks and detections
        emb_dists (numpy.ndarray): Embedding distance matrix between tracks and detections
        title (str): Title for the visualization
        crop_size (tuple): Size of crops (width, height)
        t_ids (list): List of track IDs (optional)

    Returns:
        tuple: (visualization image, cell positions, track crops, detection crops)
    """
    tracks_num = len(t_boxes)
    detections_num = len(d_boxes)
    w, h = crop_size

    if emb_dists is None:
        emb_dists = np.ones_like(iou_dists)

    if t_ids is None:
        t_ids = list(range(tracks_num))

    # New grid: rows = tracks_num + 2, columns = detections_num + 2
    target_height = tracks_num + 2
    target_width = detections_num + 2

    img_size = (h * target_height, w * target_width, 3)
    dist_img = np.full(img_size, 225, dtype='uint8')

    # Convert boxes (assuming xcycwh format to xyxy)
    t_boxes_xyxy = [xywh2xyxy_clip(box, img.shape) for box in t_boxes]
    d_boxes_xyxy = [xywh2xyxy_clip(box, img.shape) for box in d_boxes]

    # Lists to store crops and cell positions for mouse hover
    track_crops = []
    detection_crops = []
    cell_positions = []

    # --- Insert IDs ---
    # Top header: Detection IDs (for columns)
    for j in range(detections_num):
        id_str = str(j)
        conf = dets_confs[j]
        color = (0, 0, 0) if conf > 0.5 else (255, 0, 0)
        cv2.putText(dist_img, f"{id_str} ({conf:.2f})", ((j + 2) * w + 5, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # Left header: Track IDs (for rows)
    for i in range(tracks_num):
        id_str = str(t_ids[i])
        cv2.putText(dist_img, id_str, (5, (i + 2) * h + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # --- Insert Crops ---
    # Detection crops (current frame) in the top header row (row index 1)
    for j, box in enumerate(d_boxes_xyxy):
        crop = get_object_crop(img, box, crop_size)
        x0, x1 = (j + 2) * w, (j + 2) * w + w
        y0, y1 = h, 2 * h
        dist_img[y0:y1, x0:x1, :] = crop
        detection_crops.append(crop)

    # Track crops (previous frame) in the left header column (column index 1)
    for i, box in enumerate(t_boxes_xyxy):
        crop = get_object_crop(img, box, crop_size)
        y0, y1 = (i + 2) * h, (i + 2) * h + h
        x0, x1 = w, 2 * w
        dist_img[y0:y1, x0:x1, :] = crop
        track_crops.append(crop)

    # --- Insert Grid Lines ---
    # Horizontal lines (rows)
    for n, y in enumerate(range(h, dist_img.shape[0] + 1, h)):
        # For the line that separates the header (first two rows) from the main grid, start at x=2*w.
        x_start = 2 * w if n == 1 else 0
        cv2.line(dist_img, (x_start, y - 1),
                 (dist_img.shape[1] - 1, y - 1), (0, 0, 0), 1, 1)

    # Vertical lines (columns)
    for n, x in enumerate(range(w, dist_img.shape[1] + 1, w)):
        # For the line that separates the header (first two columns) from the main grid, start at y=2*h.
        y_start = 2 * h if n == 1 else 0
        cv2.line(dist_img, (x - 1, y_start),
                 (x - 1, dist_img.shape[0] - 1), (0, 0, 0), 1, 1)

    # --- Insert Hat (Top-left block) ---
    cv2.line(dist_img, (0, 0), (2 * w, 2 * h), (0, 0, 0), 1, 1)
    cv2.putText(dist_img, 'Dets', (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(dist_img, 'Tracks', (4, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # --- Insert Distance Metrics into the Main Grid ---
    # Main grid cells start at row index 2 and column index 2.
    for i in range(tracks_num):
        for j in range(detections_num):
            iou_val = str(iou_dists[i][j])[:4] if iou_dists[i][j] else '-'
            emb_val = str(emb_dists[i][j])[:4] if emb_dists[i][j] else '-'

            # Calculate top-left corner of the cell: (column, row) = ((j+2)*w, (i+2)*h)
            cell_x = (j + 2) * w
            cell_y = (i + 2) * h
            position_iou = (cell_x + 1, cell_y + 24)
            position_emb = (cell_x + 1, cell_y + 48)

            cell_positions.append({
                'track_idx': i,
                'track_id': t_ids[i],
                'det_idx': j,
                'x1': cell_x,
                'y1': cell_y,
                'x2': cell_x + w,
                'y2': cell_y + h,
                'iou': iou_val,
                'emb': emb_val,
            })

            cv2.putText(dist_img, f"I:{iou_val}", position_iou,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
            cv2.putText(dist_img, f"E:{emb_val}", position_emb,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 0, 128), 1)

    # --- Add Title Bar ---
    title_height = 32
    title_img = np.full(
        (title_height, dist_img.shape[1], 3), 225, dtype='uint8')
    cv2.putText(title_img, title, (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.line(title_img, (0, title_height - 1),
             (dist_img.shape[1] - 1, title_height - 1), (0, 0, 0), 1, 1)

    final_img = np.vstack([title_img, dist_img])

    return final_img, cell_positions, track_crops, detection_crops


def save_visualization(img, file_path):
    """
    Save visualization to file

    Args:
        img (numpy.ndarray): Visualization image
        file_path (str): Path to save the image
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(file_path, img)


def show_visualization(vis_img, cell_positions, track_crops, detection_crops,
                       window_name="Tracking_Visualization", popup_name="Detail_View",
                       window_size=(960, 720), frame_num: int = None):
    """
    Display non-blocking visualization with a popup on hover.

    Args:
        vis_img (numpy.ndarray): Visualization image.
        cell_positions (list): List of cell position dictionaries.
        track_crops (list): List of track crops.
        detection_crops (list): List of detection crops.
        window_name (str): Name of the main window.
        popup_name (str): Name of the popup window.
        window_size (tuple): Desired window size (width, height).
    """

    if frame_num is not None:
        text = f"Frame: {frame_num}"
        # Choose a position, font, scale, color, and thickness.
        cv2.putText(vis_img, text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    gui_manager.show(window_name, vis_img, window_size=window_size)

    # Set mouse callback with parameters
    param = (vis_img, cell_positions, track_crops, detection_crops, popup_name)
    # gui_manager.set_callback(window_name, mouse_callback, param)
    

    # if cv2.waitKey(1) & 0xFF != ord('q'):
    #     pass


# Example usage:
"""
import numpy as np

# Sample data
prev_img = np.zeros((720, 1280, 3), dtype=np.uint8)  # Previous frame
prev_img[100:300, 100:300] = (0, 0, 255)  # Add some red rectangle for track 0
prev_img[200:400, 500:700] = (255, 0, 0)  # Add some blue rectangle for track 1

cur_img = np.zeros((720, 1280, 3), dtype=np.uint8)   # Current frame
cur_img[110:310, 110:310] = (0, 0, 255)  # Add some red rectangle for detection 0
cur_img[190:390, 490:690] = (255, 0, 0)  # Add some blue rectangle for detection 1

t_boxes = [[100, 200, 200, 400], [500, 300, 200, 400]]  # Track boxes [xcenter, ycenter, width, height]
d_boxes = [[110, 210, 200, 400], [490, 290, 200, 400]]  # Detection boxes [xcenter, ycenter, width, height]
iou_dists = np.array([[0.8, 0.1], [0.2, 0.9]])  # IoU distances
emb_dists = np.array([[0.7, 0.3], [0.4, 0.6]])  # Embedding distances

# Generate visualization with additional data for interactive view
vis_img, cell_positions, track_crops, detection_crops = visualize_tracking(
    prev_img, t_boxes, cur_img, d_boxes, iou_dists, emb_dists, "Tracking Results")

# Display non-blocking visualization
show_visualization(vis_img, cell_positions, track_crops, detection_crops)

# Your code continues here...
print("Program continues execution while visualization is displayed")

# If you want to update multiple windows or need a refresh loop in your main code:
while True:
    # Your processing code here
    
    # Check for key press in windows without blocking
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Close all windows when done
cv2.destroyAllWindows()
"""


def hex_to_cv2(hex_color: str) -> tuple:
    """
    Convert a hex color string (e.g. "#FF3838") to a BGR tuple for OpenCV.
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


# 25-color palette for track boxes
PALETTE_25 = [
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
    "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
    "#2C99A8", "#00C2FF", "#344593", "#6473FF", "#0018EC",
    "#8438FF", "#520085", "#CB38FF", "#FF95C8", "#FF37C7",
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"
]


def plot_frame(frame_path: str,
               tracks_boxes: list[list[int]] = None,
               tracks_ids: list[int] = None,
               dets_boxes: list[list[int]] = None,
               dets_confs: list[float] = None,
               frame_shape: tuple[int, int] = None,):
    """
    Load an image from a file path and plot it with track and detection boxes,
    adding outlined numbers for better recognition.

    Parameters
    ----------
    frame_path : str
        Path to the image file.
    tracks_boxes : list[list[int]], optional
        List of track bounding boxes in [x1, y1, x2, y2] format.
    tracks_ids : list[int], optional
        List of track IDs corresponding to the track boxes.
    dets_boxes : list[list[int]], optional
        List of detection bounding boxes in [x1, y1, x2, y2] format.
    dets_confs : list[float], optional
        List of detection confidences corresponding to the detection boxes.
    """
    # Load image from disk
    im = cv2.imread(frame_path)

    if im is None:
        raise ValueError(f"Image at path '{frame_path}' could not be loaded.")
    
    if frame_shape is not None:
        im = cv2.resize(im, frame_shape)

    # Determine line thickness based on image size
    height, width = im.shape[:2]
    thickness = max(2, round(2 / 1920 * max(width, height)))

    # Define an outline color for text (dark color for contrast)
    outline_color = hex_to_cv2("#000000")  # Black outline

    # Draw track boxes and IDs
    if tracks_boxes is not None:
        for i, box in enumerate(tracks_boxes):
            # Use the provided track id if available; otherwise, fallback to index
            track_id = tracks_ids[i] if (
                tracks_ids is not None and i < len(tracks_ids)) else i
            # Select color from palette using modulo 25
            color = PALETTE_25[track_id % len(PALETTE_25)]
            cv_color = hex_to_cv2(color)
            # Draw rectangle: box format is [x1, y1, x2, y2]
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            cv2.rectangle(im, pt1, pt2, color=cv_color, thickness=thickness)
            # Draw track ID with outline for improved readability
            text_org = (pt1[0], max(pt1[1] - 10, 0))
            text = str(track_id)
            cv2.putText(im, text, text_org,
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, outline_color, 2 * thickness)
            cv2.putText(im, text, text_org,
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, cv_color, thickness)

    # Draw detection boxes and confidence numbers in a distinct color
    if dets_boxes is not None:
        DETS_COLOR = "#00CED1"  # Dark Turquoise for detections
        cv_dets_color = hex_to_cv2(DETS_COLOR)
        for i, box in enumerate(dets_boxes):
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            cv2.rectangle(im, pt1, pt2, color=cv_dets_color,
                          thickness=thickness)
            # If a confidence score is provided, add it as text with an outline
            if dets_confs is not None and i < len(dets_confs):
                conf = dets_confs[i]
                label = f"{conf:.2f}"
                text_org = (pt2[0], max(pt2[1] + 10, 0))
                cv2.putText(im, label, text_org,
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, outline_color, 2 * thickness)
                cv2.putText(im, label, text_org,
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, cv_dets_color, thickness)

    # Convert from BGR to RGB for display with matplotlib
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(im_rgb)
    plt.axis('off')
    plt.title("Frame")
    plt.show()
