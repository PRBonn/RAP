# Test script example

python sample.py --config-name "RAP_inference" \
visualizer.max_samples_per_batch=5 \
visualizer.save_trajectory=false \
model=rap_10 \
model.save_results=true \
model.inference_sampling_steps=10 \
model.n_generations=3 \
model.rigidity_forcing=true \
model.return_end_point_trajectory=true \
data.max_points_per_batch=50000 \
data.force_use_ply=true \
data.use_random_split=true \
data.dataset_names=[threedmatch,kitti,your_dataset_name_1,your_dataset_name_2] \
data.max_parts=12 \
data.min_parts=2 \
ckpt_path=./weights/rap_model.ckpt