It takes audio files from user, and extracts its features using audioset model.



# Download vggish model first.

$ curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
$ curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz

# Then run it.
python main.py

vggish codes are excepted from audioset.
https://research.google.com/audioset/download.html


Il-Young Jeong
iyjeong@cochlear.ai
