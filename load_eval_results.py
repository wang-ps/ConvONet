import pandas as pd

filename = '/mnt/logs/docnn/room_eval/all_gpu4_ep600_poly_d7_basic_reweight_in_out_mesh/eval_meshes_full.pkl'
rst = pd.read_pickle(filename)
rst.to_csv(filename + '.csv')
print('succ')
