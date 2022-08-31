# mscoco-flir
#python core/main.py \
#--tgt_cat flir --n_classes 3 --epochs 15 \
#--device cuda:1 --logdir outputs/flir
# mscoco-m3fd
python core/main.py \
--tgt_cat m3fd --n_classes 6 --epochs 30 \
--device cuda:0 --logdir outputs/m3fd
