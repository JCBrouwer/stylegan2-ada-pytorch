#!/bin/zsh

export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=online
conda activate maua

python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "Baseline" --pruning mask --lambda_l1 0 --distill self-supervised --quantization none

python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "Basic, prune: 5e-3" --pruning mask --lambda_l1 .005 --distill basic --quantization none
python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "Basic, prune: 2.5e-3" --pruning mask --lambda_l1 .0025 --distill basic --quantization none
python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "Basic, prune: 1e-3" --pruning mask --lambda_l1 .001 --distill basic --quantization none
python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "Basic, prune: 5e-4" --pruning mask --lambda_l1 .0005 --distill basic --quantization none
python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "Basic, prune: 1e-4" --pruning mask --lambda_l1 .0001 --distill basic --quantization none

python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "LPIPS VGG, prune: 5e-3" --pruning mask --lambda_l1 .005 --distill lpips --quantization none
python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "LPIPS VGG, prune: 2.5e-3" --pruning mask --lambda_l1 .0025 --distill lpips --quantization none
python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "LPIPS VGG, prune: 1e-3" --pruning mask --lambda_l1 .001 --distill lpips --quantization none
python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "LPIPS VGG, prune: 5e-4" --pruning mask --lambda_l1 .0005 --distill lpips --quantization none
python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "LPIPS VGG, prune: 1e-4" --pruning mask --lambda_l1 .0001 --distill lpips --quantization none

python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "LPIPS Alex, prune: 5e-3" --pruning mask --lambda_l1 .005 --distill lpips --lpips-net alex --quantization none
python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "LPIPS Alex, prune: 2.5e-3" --pruning mask --lambda_l1 .0025 --distill lpips --lpips-net alex --quantization none
python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "LPIPS Alex, prune: 1e-3" --pruning mask --lambda_l1 .001 --distill lpips --lpips-net alex --quantization none
python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "LPIPS Alex, prune: 5e-4" --pruning mask --lambda_l1 .0005 --distill lpips --lpips-net alex --quantization none
python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "LPIPS Alex, prune: 1e-4" --pruning mask --lambda_l1 .0001 --distill lpips --lpips-net alex --quantization none

python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "Self-supervised, prune: 5e-3" --pruning mask --lambda_l1 .005 --distill self-supervised --quantization none
python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "Self-supervised, prune: 2.5e-3" --pruning mask --lambda_l1 .0025 --distill self-supervised --quantization none
python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "Self-supervised, prune: 1e-3" --pruning mask --lambda_l1 .001 --distill self-supervised --quantization none
python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "Self-supervised, prune: 5e-4" --pruning mask --lambda_l1 .0005 --distill self-supervised --quantization none
python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "Self-supervised, prune: 1e-4" --pruning mask --lambda_l1 .0001 --distill self-supervised --quantization none

python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "Quantize prune: 1e-3" --pruning mask --lambda_l1 .001 --distill self-supervised --quantization linear
python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "Quantize prune: 5e-4" --pruning mask --lambda_l1 .0005 --distill self-supervised --quantization linear
python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "Quantize prune: 1e-4" --pruning mask --lambda_l1 .0001 --distill self-supervised --quantization linear

python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "Quantize (Synthesis Only) prune: 1e-3" --pruning mask --lambda_l1 .001 --distill self-supervised --quantization linear --quantize-mapping False
python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "Quantize (Synthesis Only) prune: 5e-4" --pruning mask --lambda_l1 .0005 --distill self-supervised --quantization linear --quantize-mapping False
python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "Quantize (Synthesis Only) prune: 1e-4" --pruning mask --lambda_l1 .0001 --distill self-supervised --quantization linear --quantize-mapping False

python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "QGAN (Synthesis Only) prune: 1e-3" --pruning mask --lambda_l1 .001 --distill self-supervised --quantization qgan --quantize-mapping False
python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "QGAN (Synthesis Only) prune: 5e-4" --pruning mask --lambda_l1 .0005 --distill self-supervised --quantization qgan --quantize-mapping False
python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "QGAN (Synthesis Only) prune: 1e-4" --pruning mask --lambda_l1 .0001 --distill self-supervised --quantization qgan --quantize-mapping False

python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "No RGB Pruning prune: 5e-3" --pruning mask --lambda_l1 .005 --distill self-supervised --quantization none --prune-torgb False
python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "No RGB Pruning prune: 2.5e-3" --pruning mask --lambda_l1 .0025 --distill self-supervised --quantization none --prune-torgb False
python train.py --metrics kid,fid,prdc --cfg paper256 --resume ffhq256 --outdir /home/hans/modelzoo/efficiency/ --data /home/hans/trainsets/ffhq256.zip --teacher-path /home/hans/datasets/ffhq256-teacher/ --mirror True --kimg 42 \
     --wbname "No RGB Pruning prune: 1e-3" --pruning mask --lambda_l1 .001 --distill self-supervised --quantization none --prune-torgb False