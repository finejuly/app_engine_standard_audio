import audioset
import os
import numpy as np
from PIL import Image
from utils import audioset
import tensorflow as tf

def wavfile_to_melfile(wav_filename, im_filename):
    input_batch = audioset.vggish_input.wavfile_to_examples(os.path.join('static',wav_filename))
    input_shape = np.shape(input_batch)
    input_image = np.reshape(input_batch,(input_shape[0]*input_shape[1],input_shape[2]))
    input_image = input_image-input_image.min()
    input_image = input_image/input_image.max()*255

    im = Image.fromarray(input_image)
    im = im.convert('RGB')
    im.save(os.path.join('static',im_filename), format="PNG")
    return input_batch

def feature_extraction(input_batch, fea_filename):
    checkpoint_path = 'utils/audioset/vggish_model.ckpt'
    pca_params_path = 'utils/audioset/vggish_pca_params.npz'
    with tf.Graph().as_default(), tf.Session() as sess:
      audioset.vggish_slim.define_vggish_slim()
      audioset.vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

      features_tensor = sess.graph.get_tensor_by_name(
          audioset.vggish_params.INPUT_TENSOR_NAME)
      embedding_tensor = sess.graph.get_tensor_by_name(
          audioset.vggish_params.OUTPUT_TENSOR_NAME)
      [embedding_batch] = sess.run([embedding_tensor],
                                   feed_dict={features_tensor: input_batch})
    #  print('VGGish embedding: ', embedding_batch[0])
    #  expected_embedding_mean = 0.131
    #  expected_embedding_std = 0.238
    #  np.testing.assert_allclose(
    #      [np.mean(embedding_batch), np.std(embedding_batch)],
    #      [expected_embedding_mean, expected_embedding_std],
    #      rtol=rel_error)

    # Postprocess the results to produce whitened quantized embeddings.
    pproc = audioset.vggish_postprocess.Postprocessor(pca_params_path)
    postprocessed_batch = pproc.postprocess(embedding_batch)
    im = Image.fromarray(postprocessed_batch)
    im = im.convert('RGB')
    im.save(os.path.join('static',fea_filename), format="PNG")
    #print('Postprocessed VGGish embedding: ', postprocessed_batch[0])
    #expected_postprocessed_mean = 123.0
    #expected_postprocessed_std = 75.0
    #np.testing.assert_allclose(
    #    [np.mean(postprocessed_batch), np.std(postprocessed_batch)],
    #    [expected_postprocessed_mean, expected_postprocessed_std],
    #    rtol=rel_error)
    return postprocessed_batch
