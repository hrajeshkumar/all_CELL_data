"""
Usage
	python create_tf_records.py \
      --image_dir="${IMAGE_DIR}" \
      --annotations_csv="${ANNOTATIONS_CSV}" \
      --output_dir="${OUTPUT_DIR}"
"""	

import tensorflow as tf
from PIL import Image
import pandas as pd
from pathlib import Path
import os 
from tqdm import tqdm
from object_detection.utils import dataset_util
from absl import app
from absl import flags


flags.DEFINE_string('image_dir', '', 'Image directory.')
flags.DEFINE_string('annotations_csv', '', 'Annotations in .csv format.')
flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')

FLAGS = flags.FLAGS

def create_tf_example(image_dir, image_file_name, annotations_image_df):
    #Load the image
    full_path = os.path.join(image_dir, image_file_name)
    image_format = Path(full_path).suffix
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    image_width, image_height = image.size

    #Create arrays to store information about all bounding boxes for this image
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    class_names = []

    #Populate attributes of bounding boxes 
    for index, row in annotations_image_df.iterrows():
        xmins.append(row['xmin'] / image_width)
        xmaxs.append(row['xmax'] / image_width)
        ymins.append(row['ymin'] / image_height)
        ymaxs.append(row['ymax'] / image_height)
        class_names.append(row['class'].encode('utf8'))

    #Create the feature dictionary
    feature_dict = {
        'image/height': dataset_util.int64_feature(image_height),
        'image/width': dataset_util.int64_feature(image_width),
        'image/filename': dataset_util.bytes_feature(image_file_name.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(image_file_name.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(class_names)
    }

    #Create the tf example
    tf_example = tf.train.Example(features=tf.train.Features(feature_dict))
    return tf_example

def create_tf_records(image_dir, annotations_csv, output_dir):
    with tf.io.TFRecordWriter(output_dir) as writer:
        all_image_files = list(os.listdir(image_dir))
        annotations_df = pd.read_csv(annotations_csv)
        missing_annotations = []
        for image_file_name in tqdm(all_image_files):
            annotations_image_df = annotations_df[annotations_df['filename'] == image_file_name]
            #It is possible that annotations are missing for some images
            if annotations_image_df.empty:
                missing_annotations.append(image_file_name)
                continue
            
            tf_example = create_tf_example(image_dir, image_file_name, annotations_image_df)
            writer.write(tf_example.SerializeToString())

    if len(missing_annotations) > 0:
        print('Annotations are missing for the following images')
        print(missing_annotations)
    
    print('Successfully created the TFRecords: {}'.format(output_dir))

def main(_):
    image_dir = os.path.join(FLAGS.image_dir)
    annotations_csv = os.path.join(FLAGS.annotations_csv)
    output_dir = os.path.join(FLAGS.output_dir)
    create_tf_records(image_dir, annotations_csv, output_dir)

if __name__ == '__main__':
    app.run(main)