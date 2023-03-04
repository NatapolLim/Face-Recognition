import torch
from dataclasses import dataclass
import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Union, Dict, Tuple, Generator, Callable

from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO

from supervision.video import VideoInfo, VideoSink, get_video_frames_generator
from supervision.draw.color import ColorPalette, Color
from supervision.detection.core import BoxAnnotator, Detections
from onemetric.cv.utils.iou import box_iou_batch

from norfair import Detection, Tracker, OptimizedKalmanFilterFactory
from norfair.tracker import TrackedObject

class VideoAnalytic(VideoInfo):
    pass

class YOLOByteTrack():
    pass

@dataclass
class PredictModel:
    method: str
    mtcnn: Optional[MTCNN] = None
    model: Union[YOLO, InceptionResnetV1] = None
    resnet: Optional[InceptionResnetV1] = None
    label_map: Optional[Dict] = None
    device: torch.device = torch.device('cpu')

    @classmethod
    def activate(cls, method: str, label_map: Optional[Dict], device: torch.device = torch.device('cpu')):
        if method=='MTCNN':
            mtcnn = MTCNN(min_face_size=100, keep_all=True,)
            print('load model to',device)
            model = torch.load('models/face_detect_v4.1_3.pt').eval().to(device)
            resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            return PredictModel(method, mtcnn=mtcnn, model=model, resnet=resnet, label_map=label_map, device=device)
        elif method=='YOLO':
            model = YOLO("models/yolov8n.pt")
            model.fuse()
            label_map = model.model.names
            return PredictModel(method=method, model=model, label_map=label_map)

    def predict(self, img: np.ndarray, ret_tensor: bool = False) -> Union[Detections, torch.Tensor]:
        if self.method=='MTCNN':
            boxes, probs = self.mtcnn.detect(img, landmarks=False)
            if boxes is not None and len(boxes)>0:
                tensor = self.mtcnn.extract(img, boxes, save_path=None)
                if self.device=='mps':
                    tensor = tensor.to(self.device)
                results = self.model(tensor).softmax(1).detach().cpu().numpy()
                idxs = np.argmax(results, axis=1)
                detections = Detections(
                    xyxy=boxes,
                    confidence=probs,
                    class_id=idxs
                    )
                if ret_tensor:
                    return detections, tensor
                return detections
            if ret_tensor:
                    return np.empty(0,), np.empty(0,)
            return np.empty(0,)
        elif self.method=='YOLO':
            results = self.model(img)
            detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int)
                )
            return detections

class NorfairReid():
    def __init__(
            self,
            video_path: str,
            output_path: str,
            method: str,
            label_map: Dict = None,
            activate_norfair: bool = True,
            device: torch.device = torch.device('cpu'),
            activate_face_embed_to_dist_func: bool = False
            ) -> None:
        
        self.video_path = video_path
        self.method = method
        self.label_map = label_map
        self.activate_norfair = activate_norfair
        self.video_output = output_path
        self.generator, self.video_info, self.box_annotator = Utils.prepare(self.video_path)
        self.activate_face_embed_to_dist_func = activate_face_embed_to_dist_func
        if activate_face_embed_to_dist_func:
            distance_func = Utils.face_and_hist_distance
        else:
            distance_func = Utils.embedding_distance
        if activate_norfair:
            self.track_box_annotator = BoxAnnotator(color=Color.blue(), thickness=2, text_thickness=2, text_scale=1)
            self.tracker = Tracker(
                initialization_delay=10,
                distance_function="euclidean",
                hit_counter_max=20,
                filter_factory=OptimizedKalmanFilterFactory(),
                distance_threshold=50,
                past_detections_length=5,
                reid_distance_function=distance_func,
                reid_distance_threshold=0.5,
                reid_hit_counter_max=500,
                )
        self.device = device
        # print(f'{self.method} Predictor on Device {self.device}')

    def process(self):
        if self.video_output is not None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_tracked = cv2.VideoWriter(
                self.video_output,
                fourcc,
                self.video_info.fps,
                (self.video_info.width, self.video_info.height)
                )

        model = PredictModel.activate(self.method, self.label_map, self.device) 
        if model.label_map is None:
            raise Exception('Label map is None')

        for i, img in tqdm(enumerate(self.generator), total=self.video_info.total_frames-1):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if self.activate_face_embed_to_dist_func:
                detections, tensors= model.predict(img, ret_tensor=True)
            else:
                detections = model.predict(img, ret_tensor=False)
            if isinstance(detections, Detections):
                norfair_detection = Connector.detections2detection(detections)
                for i, detection in enumerate(norfair_detection):
                    cut = Utils.get_cutout(detection.points, img)
                    hist = Utils.get_hist(cut)
                    if self.activate_face_embed_to_dist_func:
                        face_embedding = model.resnet(tensors[i].unsqueeze(0))
                        
                        detection.embedding = (face_embedding, hist)
                    else:
                        detection.embedding = hist
                    # if cut.shape[0] > 0 and cut.shape[1] > 0:
                    # else:
                    #     detection.embedding = None
                
                if self.activate_norfair:
                    tracked_objects = self.tracker.update(detections=norfair_detection)
                    track_detections = Connector.trackobj2detections(tracked_objects)
                    tracker_id, _ = Utils.match_detections_with_tracks(detections=detections, tracks=track_detections)
                    # print(tracker_id)
                    detections.tracker_id = np.array(tracker_id)
                    mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
                    detections.filter(mask=mask, inplace=True)
                else:
                    tracked_objects=[]
                
                
                # print(detections)
                labels = [
                    f"#{trackker_id} {model.label_map[class_id]} {confidence:0.2f}"
                    for _, confidence, class_id, trackker_id
                    in detections
                    ]
                img = self.box_annotator.annotate(scene=img, detections=detections, labels=labels)

            # if len(tracked_objects)>0:
            #     # track_id = Connector.trackobject2track_id(tracked_objects)
            #     # detections.tracker_id = np.array(track_id)
            #     track_detections = Connector.trackobj2detections(tracked_objects)
            #     track_labels = [f"Track id:{d[2]}" for d in track_detections]
            #     img = self.track_box_annotator.annotate(scene=img, detections=track_detections, labels=track_labels)

        
            video_tracked.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
        if self.video_output is not None:
            video_tracked.release()
            print(f"Save video at: {self.video_output}") 

# class MTCNNNorfair():

#     def __init__(
#             self,
#             video_path: str,
#             output_path: str,
#             label_map: Dict,
#             model_path: Optional[str],
#             disable_norfair: bool = True,
#             ) -> None:
#         self.video_path = video_path
#         self.mtcnn = MTCNN(min_face_size=100, keep_all=True)
#         if model_path is not None:
#             self.model = torch.load(model_path).eval().to(torch.device('cpu'))
#         else:
#             self.model = torch.load("models/face_detect_v4.1_3.pt").eval().to(torch.device('cpu'))
#         self.label_map = label_map
#         self.disable_norfair = disable_norfair
#         self.video_output = output_path
#         self.generator, self.video_info, self.box_annotator = Utils.prepare(self.video_path)
#         self.track_box_annotator = BoxAnnotator(color=Color.blue(), thickness=2, text_thickness=2, text_scale=1)
#         self.tracker = Tracker(
#             initialization_delay=10,
#             distance_function="euclidean",
#             hit_counter_max=20,
#             filter_factory=OptimizedKalmanFilterFactory(),
#             distance_threshold=50,
#             past_detections_length=5,
#             reid_distance_function=Utils.embedding_distance,
#             reid_distance_threshold=0.5,
#             reid_hit_counter_max=1000,
#             )

#     def process(self) -> None:
#         if self.video_output is not None:
#             fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#             video_tracked = cv2.VideoWriter(
#                 self.video_output,
#                 fourcc,
#                 self.video_info.fps,
#                 (self.video_info.width, self.video_info.height)
#                 )
        
#         for i, img in tqdm(enumerate(self.generator), total=self.video_info.total_frames-1):
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             boxes, probs = self.mtcnn.detect(img, landmarks=False)
#             if boxes is not None and len(boxes)>0:
#                 tensor = self.mtcnn.extract(img, boxes, save_path=None)
#                 results = self.model(tensor).softmax(1).detach().numpy()
#                 idxs = np.argmax(results, axis=1)
#                 detections = Detections(
#                     xyxy=boxes,
#                     confidence=probs,
#                     class_id=idxs
#                     )
                
#                 norfair_detection = Connector.detections2detection(detections)
#                 for detection in norfair_detection:
#                     #detection.points = [[x1,y1],[x2,y2]] be this form
#                     cut = Utils.get_cutout(detection.points, img)
#                     # cut = get_cutout(detection.points, frame)
#                     if cut.shape[0] > 0 and cut.shape[1] > 0:
#                         detection.embedding = Utils.get_hist(cut)
#                     else:
#                         detection.embedding = None

#                 tracked_objects = self.tracker.update(detections=norfair_detection)
#                 labels = [
#                     f"{self.label_map[idx]} {confidence:0.2f}"
#                     for _, confidence, idx, _
#                     in detections
#                     ]
                
#                 img = self.box_annotator.annotate(
#                     scene=np.array(img),
#                     detections=detections,
#                     labels=labels
#                     )
                
#                 if len(tracked_objects)>0:
#                     # track_id = Connector.trackobject2track_id(tracked_objects)
#                     # detections.tracker_id = np.array(track_id)
#                     track_detections = Connector.trackobj2detections(tracked_objects)
#                     track_labels = [f"Track id:{d[2]}" for d in track_detections]
#                     img = self.track_box_annotator.annotate(scene=img, detections=track_detections, labels=track_labels)
#             video_tracked.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

#         if self.video_output is not None:
#             video_tracked.release()

#         print(f"Save video at: {self.video_output}")


class Utils:
    @staticmethod
    def prepare(video_path: str, color: ColorPalette=ColorPalette.default()) -> Tuple[Generator, VideoInfo, BoxAnnotator]:
        generator = get_video_frames_generator(video_path)
        video_info = VideoInfo.from_video_path(video_path)
        box_annotator = BoxAnnotator(color=color, thickness=2, text_thickness=2, text_scale=1)
        print('Set generator, video info, box annotator')
        return generator, video_info, box_annotator
    
    @staticmethod
    def get_hist(image):
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        return hist
    
    @staticmethod
    def get_cutout(points, image, margin=0):
        """Returns a rectangular cut-out from a set of points on an image"""
        max_x = int(max(points[:, 0])-margin)
        min_x = int(min(points[:, 0])+margin)
        max_y = int(max(points[:, 1])-margin)
        min_y = int(min(points[:, 1])+margin)
        return image[min_y:max_y, min_x:max_x]
    
    
    @staticmethod
    def face_and_hist_distance(matched_not_init_trackers, unmatched_trackers):
        snd_embedding = unmatched_trackers.last_detection.embedding
        if snd_embedding is None:
            for detection in reversed(unmatched_trackers.past_detections):
                if detection.embedding is not None:
                    snd_embedding = detection.embedding
                    break
            else:
                return 1

        for detection_fst in matched_not_init_trackers.past_detections:
            if detection_fst.embedding is None:
                continue
            snd_face_embed, snd_hist = snd_embedding
            fst_face_embed, fst_hist = detection_fst.embedding

            hist_distance = 1 - cv2.compareHist(
                snd_hist, fst_hist, cv2.HISTCMP_CORREL
            )

            em_distance = (fst_face_embed-snd_face_embed).norm().item()
            # normalize to scale 0-1
            em_distance = min(1.5, em_distance)/1.5
            # print(em_distance, hist_distance)
            ratio=0.5
            summ = em_distance*(ratio) + hist_distance*(1-ratio)
            # print('sum:', summ)
            # if summ < 0.5:
            return summ
        return 1
    
    @staticmethod
    def embedding_distance(matched_not_init_trackers, unmatched_trackers):
        snd_embedding = unmatched_trackers.last_detection.embedding
        if snd_embedding is None:
            for detection in reversed(unmatched_trackers.past_detections):
                if detection.embedding is not None:
                    snd_embedding = detection.embedding
                    break
            else:
                return 1

        for detection_fst in matched_not_init_trackers.past_detections:
            if detection_fst.embedding is None:
                continue

            distance = 1 - cv2.compareHist(
                snd_embedding, detection_fst.embedding, cv2.HISTCMP_CORREL
            )
            if distance < 0.5:
                return distance
        return 1
    
    def match_detections_with_tracks(detections: Detections, tracks: Detections) -> Tuple[List, Detections]:
        if not np.any(detections.xyxy) or tracks is None:
            return np.empty((0,)), np.empty((0,))
    
        iou = box_iou_batch(tracks.xyxy, detections.xyxy)
        # print('iou',iou)
        track2detection = np.argmax(iou, axis=1)
        # print('argmax',track2detection)
        tracker_ids = [None] * len(detections)
        
        for tracker_index, detection_index in enumerate(track2detection):
            # print('select',iou[tracker_index, detection_index])
            if iou[tracker_index, detection_index] != 0:
                # print('track_class_id', tracks.class_id[tracker_index])
                tracker_ids[detection_index] = tracks.class_id[tracker_index]

        return tracker_ids, tracks

class Connector:
    @staticmethod
    def detections2detection(detections: Detections) -> Detection:
        # if not isinstance(detections, Detections):
        #     return np.empty(0,)
        norfair_detection=[]
        for xyxy, _, _, _ in detections:
            box = xyxy
            # conf = super_de.confidence
            nor_de = Detection(
                points=np.array([[box[0],box[1]],[box[2],box[3]]])
                )
            norfair_detection.append(nor_de)
        return norfair_detection
    
    @staticmethod
    def trackobj2detections(tracked_objects: List[TrackedObject]) -> Detections:
        xyxy =[]
        conf = np.ones(len(tracked_objects))
        track_id = []
        for obj in tracked_objects:
            box = obj.estimate
            box = np.array([box[0][0],box[0][1],box[1][0],box[1][1]], dtype=int)
            xyxy.append(box)
            track_id.append(obj.id)

        if len(xyxy)==0:
            return None
        
        track_detections = Detections(
                xyxy = np.array(xyxy),
                confidence=conf,
                class_id= np.array(track_id)
                )
        return track_detections
    
    @staticmethod
    def trackobject2track_id(tracked_objects: List[TrackedObject]) -> List[int]:
        return [obj.id for obj in tracked_objects]
    