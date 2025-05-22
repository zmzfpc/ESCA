num=$(od -An -N2 -i /dev/urandom | tr -d ' ')
port_num=$((num % 40000 + 10000))

echo "Random port: $port_num"
threshold=80
sym=False
model_calib=False
mask=True
mask_weighted=True
act_order=True
test_new_arch=False
omni=False
max_iter=20000
lr=3e-5
reversed=False
wbit=8
abit=8
groupsize=4
colmn=False
row=False
arch="warp"
pixel=False
use2d=False
train=False
cr=0.9
echo "cr ${cr}"
echo "Running PTQ with arch ${arch} wbit ${wbit} abit ${abit} groupsize ${groupsize} mask ${mask} mask_weighted ${mask_weighted} test_new_arch ${test_new_arch} act_order ${act_order} omni ${omni} max_iter ${max_iter} threshold ${threshold} model_calib ${model_calib} lr ${lr} reversed ${reversed}"
echo "sym ${sym}"
main_path="/scratch/{fold_id}/ESCA/"


export C_INCLUDE_PATH=$CONDA_PREFIX/include:$C_INCLUDE_PATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH

python -m torch.distributed.launch --nproc_per_node=1 --master_port ${port_num} quantize.py \
    --data_dir /vast/{fold_id}/multiface/m--20180426--0000--002643814--GHS \
    --krt_dir  /vast/{fold_id}/multiface/m--20180426--0000--002643814--GHS/KRT \
    --framelist_train  /vast/{fold_id}/multiface/m--20180426--0000--002643814--GHS/frame_list.txt \
    --framelist_test  /vast/{fold_id}/multiface/m--20180426--0000--002643814--GHS/frame_list.txt \
    --model_ckpt "./pretrained_model/002643814/${arch}/best_model.pth" \
    --arch ${arch} \
    --n_worker 1 \
    --wbit ${wbit} \
    --abit ${abit} \
    --tau ${threshold} \
    --num_samples 512 \
    --model_calib ${model_calib} \
    --train_batch_size 8 \
    --result_path "./runs/experiment_002643814/PTQ_${arch}_w${wbit}a${abit}_${groupsize}_tau${threshold}_model_calib${model_calib}_mask${mask}_maskW${mask_weighted}_M_GPTQ_Omni${omni}_lr${lr}_S${sym}_P${pixel}_U2d${use2d}_CR${cr}_${train}_TN${test_new_arch}/" \
    --act-order ${act_order}\
    --wgroupsize ${groupsize} \
    --mask ${mask}\
    --mask-weighted ${mask_weighted} \
    --omni ${omni} \
    --max_iter ${max_iter} \
    --lr ${lr} \
    --reversed ${reversed} \
    --colmn ${colmn} \
    --row ${row} \
    --new-arch ${test_new_arch}\
    --use_2d_bound ${use2d} \
    --clip_ratio ${cr} \
    --sym ${sym} \
    --train ${train} 




batch_size=12

report_psnr=True
save_tensor=False

kl_lambda=1.0

let port_num+=1
for viewID in 400029 400012 400017
do

model_path="/scratch/{fold_id}/ESCA/runs/experiment_002643814/PTQ_${arch}_w${wbit}a${abit}_${groupsize}_tau${threshold}_model_calib${model_calib}_mask${mask}_maskW${mask_weighted}_M_GPTQ_Omni${omni}_lr${lr}_S${sym}_P${pixel}_U2d${use2d}_CR${cr}_${train}_TN${test_new_arch}/model.pth"
sample_method="W${wbit}A${abit}"
calib_method="W${wbit}A${abit}_SELF"
merged_video_name="W${wbit}A${abit}_${groupsize}_mask${mask}_maskW${mask_weighted}_${viewID}_GPTQ_lr${lr}_Omni${omni}_S${sym}_P${pixel}_U2d${use2d}_CR${cr}_${train}_TN${test_new_arch}.mp4"
video_path="./runs/experiment_002643814/${viewID}/videos_${arch}_w${wbit}a${abit}_${groupsize}_${threshold}_mask${mask}_maskW${mask_weighted}_M_GPTQ_Omni${omni}_lr${lr}_S${sym}_P${pixel}_U2d${use2d}_CR${cr}_${train}_TN${test_new_arch}/"
result_path="./runs/experiment_002643814/${viewID}/visual_${arch}_w${wbit}a${abit}_${groupsize}_${threshold}_mask${mask}_maskW${mask_weighted}_M_GPTQ_Omni${omni}_lr${lr}_S${sym}_P${pixel}_U2d${use2d}_CR${cr}_${train}_TN${test_new_arch}/"
err_path="./runs/results_002643814/${viewID}/error_${arch}_w${wbit}a${abit}_${groupsize}_${threshold}_mask${mask}_maskW${mask_weighted}_M_GPTQ_Omni${omni}_lr${lr}_S${sym}_P${pixel}_U2d${use2d}_CR${cr}_${train}_TN${test_new_arch}/"
tensor_path="./runs/results_002643814/${viewID}/pred_tensor_${arch}_w${wbit}a${abit}_mask${mask}_maskW${mask_weighted}_M_GPTQ_Omni${omni}_lr${lr}_S${sym}_P${pixel}_U2d${use2d}_CR${cr}_${train}_TN${test_new_arch}/"

for i in {1..7}
do
python -m torch.distributed.launch --nproc_per_node=1 --master_port  ${port_num} visualize.py \
    --data_dir /vast/{fold_id}/multiface/m--20180426--0000--002643814--GHS \
    --krt_dir /vast/{fold_id}/multiface/m--20180426--0000--002643814--GHS/KRT \
    --framelist_test /vast/{fold_id}/multiface/m--20180426--0000--002643814--GHS/frame_list.txt \
    --result_path ${result_path} \
    --video_path ${video_path} \
    --err_path ${err_path} \
    --tensor_path ${tensor_path} \
    --err_file "frame_wise_error_${i}.csv" \
    --test_segment "/scratch/{fold_id}/ESCA/test_segments_002643814/mini_test_segment_group${i}.json" \
    --lambda_screen 1 \
    --model_path  ${model_path} \
    --camera_config "/scratch/{fold_id}/ESCA/camera_configs/camera-split-config_002643814.json" \
    --camera_setting "full" \
    --arch ${arch} \
    --val_batch_size ${batch_size} \
    --sample_method ${sample_method} \
    --calib_method ${calib_method} \
    --wbit ${wbit} \
    --abit ${abit} \
    --report_psnr ${report_psnr} \
    --save_tensor ${save_tensor} \
    --n_worker 1 \
    --view_id ${viewID} \
    --omni ${omni} \
    --colmn ${colmn} \
    --row ${row} \
    --use_2d_bound ${use2d} \
    --sym ${sym} \
    --clip_ratio ${cr} \

done

cd ${video_path}; find *.mp4 | sed 's:\ :\\\ :g'| sed 's/^/file /' > fl.txt; ffmpeg -f concat -i fl.txt -c copy ${merged_video_name}; rm fl.txt; cd ..

cd ${main_path}

python report_metrics.py \
    --metrics_path ${err_path} 

done