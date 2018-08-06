import cPickle as pkl
import h5py
from tqdm import tqdm
import os

train_h5_path = '../data/train36.hdf5'
val_h5_path = '../data/val36.hdf5'
train_id_path = '../data/train36_imgid2idx.pkl'
val_id_path = '../data/val36_imgid2idx.pkl'

save_obj_path = '../data/coco_obj'
train_obj_path = os.path.join(save_obj_path, 'train_obj.pkl')
val_obj_path = os.path.join(save_obj_path, 'val_obj.pkl')
if not os.path.exists(save_obj_path):
    os.makedirs(save_obj_path)

with open(train_id_path, 'r') as f:
    train_id = pkl.load(f)

with open(val_id_path, 'r') as f:
    val_id = pkl.load(f)

train_h5 = h5py.File(train_h5_path, 'r')
val_h5 = h5py.File(val_h5_path, 'r')

train_bb = train_h5['image_bb']
val_bb = val_h5['image_bb']

train_bb_dict = {}
for each_id, each_index in train_id.iteritems():
    train_bb_dict[each_id] = train_bb[each_index]

val_bb_dict = {}
for each_id, each_index in val_id.iteritems():
    val_bb_dict[each_id] = val_bb[each_index]

assert len(train_bb_dict.keys()) == train_bb.shape[0]
assert len(val_bb_dict.keys()) == val_bb.shape[0]

with open(train_obj_path, 'wb') as f:
    pkl.dump(train_bb_dict, f)

with open(val_obj_path, 'wb') as f:
    pkl.dump(val_bb_dict, f)