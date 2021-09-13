# Scripts to run

#python extract_embeddings.py --dataset dataset --embeddings output/PyPower_embed.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7
#python train_model.py --embeddings output/PyPower_embed.pickle --recognizer output/PyPower_recognizer.pickle --le output/PyPower_label.pickle
#python recognize_video.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/PyPower_recognizer.pickle --le output/PyPower_label.pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to output label encoder")
args = vars(ap.parse_args())

print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# write the actual face recognition model to local storage
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to local disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()