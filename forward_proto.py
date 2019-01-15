import tensorflow as tf
import numpy as np
import cv2

from IPython import embed
import glob
import os
import time
import _pickle as pickle
import humanfriendly

from hypotheses_pb2 import HypothesisSet
import config
from eval import detect_one_image


def forward_protobuf(pred_func, output_folder, forward_dataset, generic_images_folder, generic_images_pattern):
    tf.gfile.MakeDirs(output_folder)
    
    # Open each of the protobuf files we can find in turn
    # First we have to find the protobuf files for the images.
    video_str_list = [str for str in generic_images_folder.split("/") if str.startswith("vid_")]
    assert len(video_str_list) == 1, "Path structure is wrong. Path elements that start with vid: {}".format(", ".join(video_str_list))
    video_str = video_str_list[0]
    print(generic_images_folder)
    #tracking_res_folder = os.path.join("/".join(generic_images_folder.split("/")[:-2]), "tracking_results_two_heads", "pure_results", video_str)
    tracking_res_folder = os.path.join("/".join(generic_images_folder.split("/")[:-2]), "tracking_results", "pure_results", video_str)
    tracking_proto_file_list = sorted(glob.glob(os.path.join(tracking_res_folder, "*.hypset")))
    print("Found {} tracking result protobuf files in folder {}".format(len(tracking_proto_file_list), tracking_res_folder))

    # Go through each of the files and extract the features for the bounding boxes
    fps = 1.0
    for proto_idx, proto_path in enumerate(tracking_proto_file_list):
      # Do time benchmarking
      begin_time = time.time()      

      print("\n" + " Processing protobuf file {} of {} ".format(proto_idx + 1, len(tracking_proto_file_list)).center(80, "-"))
      print("path = {}".format(proto_path))
      with open(proto_path, "rb") as proto_file:
        hypo_set = HypothesisSet()
        hypo_set.ParseFromString(proto_file.read())
      
      # Create a list of features and tags for this protobuf file
      proto_tag_list = []
      proto_feature_list = []
      
      # Generate the filename for the output file. We do this early so we can skip the file if it has been processed already.
      hypo_num = len(hypo_set.hypotheses)
      bbox_num = sum([len([box for box in hyp.bounding_boxes_2D_with_timestamps]) for hyp in hypo_set.hypotheses])
      timestamp_list_list = [[timestamp for (timestamp, box) in hyp.bounding_boxes_2D_with_timestamps.items()] for hyp in hypo_set.hypotheses]
      min_frame = min([min(t_list) for t_list in timestamp_list_list])
      max_frame = max([max(t_list) for t_list in timestamp_list_list])
      out_filename = "features_maskrcnn_{}_frame_{}_to_{}__{}hyp_{}bb.pkl".format(video_str, min_frame, max_frame, hypo_num, bbox_num)
      out_path = os.path.join(output_folder, out_filename)
      if os.path.exists(out_path):
        print("Output file {} already exists. Skipping feature creation.".format(out_path))
        continue
      
      # We want to process the images in such a way that each frame is only run through the detector once. Otherwise, we are wasting time and GPU resources.
      frame_num = max_frame - min_frame + 1
      for frame_idx, frame_timestamp in enumerate(range(min_frame, max_frame + 1)):
        print("\nHandling frame {} ({} of {}, from {} to {})".format(frame_timestamp, frame_idx + 1, frame_num, min_frame, max_frame))

        # Collect the bounding boxes for this frame
        bbox_hyp_list = []
        for hyp in hypo_set.hypotheses:
          if frame_timestamp in hyp.bounding_boxes_2D_with_timestamps:
            bbox = hyp.bounding_boxes_2D_with_timestamps[frame_timestamp]
            bbox_corners = [bbox.x0, bbox.y0, bbox.x0 + bbox.w, bbox.y0 + bbox.h]
            bbox_hyp_list.append((bbox_corners, hyp))
        
        # Now, we have a list of bounding boxes paired together with the hypotheses they belong to.
        # Let's detect features for every box
        
        if len(bbox_hyp_list) == 0:
          print("Continuing with the next frame since there are no bounding boxes for frame {}".format(frame_timestamp))
          continue

        # Open the image
        img_folder = os.path.join("/".join(generic_images_folder.split("/")[:-2]), "videos", video_str, "frames_cropped", "all_frames")
        img_glob_path = os.path.join(img_folder, "*frames{:09}.png".format(frame_timestamp))
        img_file_list = glob.glob(img_glob_path)
        assert len(img_file_list) == 1, "Correct image file not found: Glob path = {}, found files = {}".format(img_glob_path, ", ".join(img_file_list))
        img_cv2 = cv2.imread(img_file_list[0], cv2.IMREAD_COLOR)
        print("Loaded image from path {}. Image shape is {}".format(img_file_list[0], img_cv2.shape))
       
        # Get the features for this image
        # We have to be conservative with GPU memory. We can't run the feature extraction for more than a handful of bounding boxes at the same time.
        # Let's split up the list of bounding box and hypo pairs into shorter lists
        #max_bbox_detected_at_once = 8
        max_bbox_detected_at_once = 400
        bbox_hyp_list_list = [bbox_hyp_list[x:x + max_bbox_detected_at_once] for x in range(0, len(bbox_hyp_list), max_bbox_detected_at_once)]
        feature_list = []
        for bbox_hypo_batch in bbox_hyp_list_list:
          input_boxes = np.array([bbox for (bbox, hyp) in bbox_hypo_batch], dtype=np.float32)
          # print("Running feature extraction for {} bounding boxes for frame {}".format(len(bbox_hypo_batch), frame_timestamp))
          detection_result_list = detect_one_image(img_cv2, pred_func, input_boxes)
          #print(len(bbox_hypo_batch), len(detection_result_list))
          feature_list += [det_result.feature_fastrcnn_pooled for det_result in detection_result_list]
        print("Detected features for {} bounding boxes for frame {} in {} batch(es).".format(len(feature_list), frame_timestamp, len(bbox_hyp_list_list)))        

        time_left_total = 500 * (len(tracking_proto_file_list) - proto_idx) * (1/fps)
        time_left_this_file = (frame_num - frame_idx - 1) * (1/fps)
        print("Time left for this file (appr.): {}".format(humanfriendly.format_timespan(time_left_this_file)))
        print("Time left total (appr.): {}".format(humanfriendly.format_timespan(time_left_total)))

        # Generate the tag for each of the bounding boxes
        # Tag format: seq_str + "___" + str(hyp_id) + "___" + str(frame_timestamp) + "___" + class_name + "___" + label
        seq_str = video_str.replace("vid_", "")
        tag_list = []
        for _, hyp in bbox_hyp_list:
          hyp_id = hyp.id
          class_name = hyp.category_name
          label = ""  # TODO: For loading groundtruth from protobuf file: label = hyp.annotated_category
          tag = seq_str + "___" + str(hyp_id) + "___" + str(frame_timestamp) + "___" + class_name + "___" + label
          tag_list.append(tag)
        
        if len(tag_list) == len(feature_list):
          # Add the tags and the features to this file's list
          proto_tag_list += tag_list
          proto_feature_list += feature_list
        else:
          print("Warning, number of bounding boxes and extracted features does not match! skipping frame!")
        #assert len(proto_tag_list) == len(proto_feature_list), "The tag list has a different length ({}) than the feature list ({}) after adding {} new tags and {} new features!".format(len(proto_tag_list), len(proto_feature_list), len(tag_list), len(feature_list))

      # This proto file is done.
      
      # Build the structure expected by the other steps in the pipeline
      pick_dic = {}
      pick_dic["tags"] = proto_tag_list
      pick_dic["ys"] = np.stack(proto_feature_list, axis=0)
      print("For this file: {} tags and bounding box features".format(len(proto_tag_list)))
      
      # Write the file
      with open(out_path, "wb") as output_file:
        pickle.dump(pick_dic, output_file, protocol=2)  # Very important: Save as python 2 pickle file!
        print("Wrote output file to {}".format(out_path))
      
      # Display timing info
      duration = time.time() - begin_time
      fps = frame_num / duration
      print("Took {} in total for this file, {} per frame, {} frames per second".format(humanfriendly.format_timespan(duration), humanfriendly.format_timespan(1/fps), fps))
    
    print("Process completed for all specified proto files.")
