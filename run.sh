python3 main.py --net_type vgg16 --version vanilla --img_resize 32  --cuda_device 0 --beta 0.0 --cutmix_prob 0.0
#python3 main.py --net_type vgg16 --version cutmix --img_resize 32  --cuda_device 0 --beta 1.0 --cutmix_prob 0.5

#python3 main.py --net_type vgg16_resizer --version vanilla --img_resize 32 --resizer_img_resize 224 --cuda_device 0 --beta 0.0 --cutmix_prob 0.0
#python3 main.py --net_type vgg16_resizer --version cutmix --img_resize 32 --resizer_img_resize 224 --cuda_device 0 --beta 1.0 --cutmix_prob 0.5

#python3 main.py --net_type vgg16_resizer_att --version vanilla --img_resize 32 --resizer_img_resize 224 --cuda_device 0 --beta 0.0 --cutmix_prob 0.0
#python3 main.py --net_type vgg16_resizer_att --version cutmix --img_resize 32 --resizer_img_resize 224 --cuda_device 0 --beta 1.0 --cutmix_prob 0.5

