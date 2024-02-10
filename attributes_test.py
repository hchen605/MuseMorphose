from utils import pickle_load

import os, pickle
import numpy as np
from collections import Counter

data_dir = 'remi_dataset'
polyph_out_dir = 'remi_dataset/attr_cls/polyph'
rhythm_out_dir = 'remi_dataset/attr_cls/rhythm'

rhym_intensity_bounds = [0.2, 0.25, 0.32, 0.38, 0.44, 0.5, 0.63]
polyphonicity_bounds = [2.63, 3.06, 3.50, 4.00, 4.63, 5.44, 6.44]

def compute_polyphonicity(events, n_bars):
  poly_record = np.zeros( (n_bars * 16,) )

  cur_bar, cur_pos = -1, -1
  for ev in events:
    if ev['name'] == 'Bar':
      cur_bar += 1
    elif ev['name'] == 'Beat':
      cur_pos = int(ev['value'])
    elif ev['name'] == 'Note_Duration':
      duration = int(ev['value']) // 120
      st = cur_bar * 16 + cur_pos
      poly_record[st:st + duration] += 1
  
  return poly_record

def get_onsets_timing(events, n_bars):
  onset_record = np.zeros( (n_bars * 16,) )

  cur_bar, cur_pos = -1, -1
  for ev in events:
    if ev['name'] == 'Bar':
      cur_bar += 1
    elif ev['name'] == 'Beat':
      cur_pos = int(ev['value'])
    elif ev['name'] == 'Note_Pitch':
      rec_idx = cur_bar * 16 + cur_pos
      onset_record[ rec_idx ] = 1

  return onset_record

if __name__ == "__main__":
  pieces = [p for p in sorted(os.listdir(data_dir)) if '.pkl' in p]

  bar_pos, events = pickle_load(os.path.join(data_dir, '1.pkl'))
  events = events[ :bar_pos[-1] ]

  #print(bar_pos[:13])#[0, 54, 105, 143, 188, 247, 302, 361, 414, 465, 530, 587, 652]
  print(events[:133])#[{'name': 'Bar', 'value': None}, {'name': 'Beat', 'value': 0}, {'name': 'Chord', 'value': 'N_N'}, {'name': 'Tempo', 'value': 119}, {'name': 'Note_Pitch', 'value': 45}, {'name': 'Note_Velocity', 'value': 60}, {'name': 'Note_Duration', 'value': 1440}, {'name': 'Beat', 'value': 2},
  #beat 0~15
  polyph_raw = np.reshape(
    compute_polyphonicity(events, n_bars=len(bar_pos)), (-1, 16)
  )
  rhythm_raw = np.reshape(
    get_onsets_timing(events, n_bars=len(bar_pos)), (-1, 16)
  )

  #print(polyph_raw[:5])
  #print(rhythm_raw[:5])
#   [[1. 1. 2. 2. 3. 4. 4. 4. 4. 4. 4. 5. 2. 1. 3. 4.]
#  [5. 6. 6. 6. 6. 6. 6. 6. 5. 2. 3. 4. 4. 5. 5. 5.]
#  [5. 5. 5. 5. 2. 1. 3. 3. 4. 4. 5. 5. 5. 6. 6. 6.]
#  [6. 6. 6. 5. 1. 2. 2. 2. 3. 3. 4. 4. 7. 7. 7. 7.]
#  [3. 2. 3. 3. 6. 6. 7. 7. 3. 3. 4. 4. 6. 5. 6. 6.]]
# [[1. 0. 1. 0. 1. 1. 0. 1. 1. 1. 0. 1. 1. 0. 1. 1.]
#  [1. 1. 0. 1. 1. 0. 1. 0. 1. 0. 1. 1. 1. 1. 0. 1.]
#  [1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 1. 0. 1. 0. 0. 1. 0. 1. 0. 1. 0. 1. 0.]
#  [1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0.]]
  polyph_cls = np.searchsorted(polyphonicity_bounds, np.mean(polyph_raw, axis=-1)).tolist()
  rfreq_cls = np.searchsorted(rhym_intensity_bounds, np.mean(rhythm_raw, axis=-1)).tolist()

  #print(polyph_cls[:5])
  #print(rfreq_cls[:5])
# [1, 5, 4, 4, 4]
# [7, 7, 4, 3, 5]
