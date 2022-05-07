import tensorflow as tf

def get_boxes(detection, outputs_shape):
    x_up = tf.dtypes.cast(tf.math.reduce_max(tf.math.argmax(detection, 0), 0), dtype = tf.float32) - (outputs_shape[1]-1)/2
    y_up = tf.dtypes.cast(tf.math.reduce_max(tf.math.argmax(detection, 1), 0), dtype = tf.float32) - (outputs_shape[0]-1)/2
    x_down = x_up + outputs_shape[1]
    y_down = y_up + outputs_shape[0]

    inputs_shape = detection.get_shape().as_list()
    
    x_up = tf.expand_dims(tf.dtypes.cast(x_up, dtype = tf.float32) / (inputs_shape[0] - 1), axis=0)
    y_up = tf.expand_dims(tf.dtypes.cast(y_up, dtype = tf.float32) / (inputs_shape[1] - 1), axis=0)
    x_down = tf.expand_dims(tf.dtypes.cast(x_down - 1, dtype = tf.float32) / (inputs_shape[0] - 1), axis=0)
    y_down = tf.expand_dims(tf.dtypes.cast(y_down - 1, dtype = tf.float32) / (inputs_shape[1] - 1), axis=0)
    return tf.transpose(tf.concat([x_up, y_up, x_down, y_down], axis = 0))


class Cropping2D_custom(tf.keras.layers.Layer):
    def __init__(self, outputs_shape, channel_detection):
        super(Cropping2D_custom, self).__init__()
        self.outputs_shape = [outputs_shape[1], outputs_shape[0]]
        self.channel_detection = channel_detection
        # self.box = tf.keras.metrics.MeanTensor(name='box')

    @tf.function
    def call(self, inputs):
        boxes = tf.map_fn(lambda x: get_boxes(x[:,:,self.channel_detection], self.outputs_shape), inputs, dtype=tf.float32)
        box_indices = tf.range(inputs.get_shape().as_list()[0])
        # self.box.update_state(boxes)
        return tf.image.crop_and_resize(inputs, boxes, box_indices, self.outputs_shape)

layer = Cropping2D_custom((3,3),3)



"""
import tensorflow as tf

def get_crops_values(detection, outputs_shape):
    x_crop = tf.math.reduce_max(tf.math.argmax(detection, 0), 0) - (outputs_shape[0]-1)//2
    y_crop = tf.math.reduce_max(tf.math.argmax(detection, 1), 0) - (outputs_shape[1]-1)//2
    return(x_crop, y_crop, outputs_shape[0], outputs_shape[1])

def crop_image(img, crop):
    # print("img:", img)        
    # ses sess.as_default():
        # print("crop[0]:", crop[0].numpy())
        # print("crop[1]:", int(crop[1].numpy()))
        # print("crop[2]:", crop[2].numpy())
        # print("crop[3]:", crop[3].numpy())
 
    # r = tf.image.crop_to_bounding_box(img, crop[0], crop[1], crop[2], crop[3])
    # print(r)
    return tf.image.crop_to_bounding_box(img, crop[0], crop[1], crop[2], crop[3])

class Cropping2D_custom(tf.keras.layers.Layer):
    def __init__(self, outputs_shape, channel_detection):
        super(Cropping2D_custom, self).__init__()
        self.outputs_shape = outputs_shape
        self.channel_detection = channel_detection

    # @tf.function
    def call(self, inputs):
        crops_values = tf.map_fn(lambda x: get_crops_values(x[:,:,self.channel_detection], self.outputs_shape), inputs, dtype=(tf.int64, tf.int64, tf.int64, tf.int64))
        # print(crops_values)
        return tf.map_fn(lambda x: crop_image(x[0], x[1]), elems=(inputs, crops_values), dtype=inputs.dtype)

layer = Cropping2D_custom((3,3),3)
"""