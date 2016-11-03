# Lightened_cnn origin align method
#python idverif.py --idimage-align /home/robert/myCoding/suresecure/dataset/20160913-ren-zheng-bi-dui-haoyun/id_frontal_data/aligned_id --normalimage-align /home/robert/myCoding/suresecure/dataset/20160913-ren-zheng-bi-dui-haoyun/id_frontal_data/aligned_frontal_xcq --model-prefix ../model/lightened_cnn/lightened_cnn --epoch 166

# Lightened_cnn mxnet align method
#python idverif.py --idimage-align /home/robert/myCoding/suresecure/dataset/20160913-ren-zheng-bi-dui-haoyun/id_frontal_data/align_mxnet/id --normalimage-align /home/robert/myCoding/suresecure/dataset/20160913-ren-zheng-bi-dui-haoyun/id_frontal_data/align_mxnet/frontal_xcq --model-prefix ../model/lightened_cnn/lightened_cnn --epoch 166

# New ID images
#python idverif.py --idimage-align /home/robert/myCoding/suresecure/dataset/20160913-ren-zheng-bi-dui-haoyun/new_id/align_mxnet/id --normalimage-align /home/robert/myCoding/suresecure/dataset/20160913-ren-zheng-bi-dui-haoyun/new_id/align_mxnet/normal --model-prefix ../model/lightened_cnn/lightened_cnn --epoch 166

# Extract feature by caffe: convert model
rm test_lfw.txt id.txt normal.txt dis.txt
python ~/myCoding/caffe/deploy/lightencnn.py --model_prototxt ../model/lightened_cnn/LightenedCNN_B_deploy.prototxt --model_file ../model/lightened_cnn/lightened_cnn_0166.caffemodel --dump_file ./test_lfw.txt /home/robert/myCoding/suresecure/dataset/20160913-ren-zheng-bi-dui-haoyun/id_frontal_data/aligned_lfw_test 
#python ~/myCoding/caffe/deploy/lightencnn.py --model_prototxt ../model/lightened_cnn/LightenedCNN_B_deploy.prototxt --model_file ../model/lightened_cnn/lightened_cnn_0166.caffemodel --dump_file ./id.txt ~/myCoding/suresecure/dataset/20160913-ren-zheng-bi-dui-haoyun/id_frontal_data/align_mxnet/id
#python ~/myCoding/caffe/deploy/lightencnn.py --model_prototxt ../model/lightened_cnn/LightenedCNN_B_deploy.prototxt --model_file ../model/lightened_cnn/lightened_cnn_0166.caffemodel --dump_file ./normal.txt ~/myCoding/suresecure/dataset/20160913-ren-zheng-bi-dui-haoyun/id_frontal_data/align_mxnet/frontal_xcq
#python idverif.py --idimage-align /home/robert/myCoding/suresecure/dataset/20160913-ren-zheng-bi-dui-haoyun/id_frontal_data/align_mxnet/id --normalimage-align /home/robert/myCoding/suresecure/dataset/20160913-ren-zheng-bi-dui-haoyun/id_frontal_data/align_mxnet/frontal_xcq --model-prefix ../model/lightened_cnn/lightened_cnn --epoch 166

## Extract feature by caffe: origin model
#rm test_lfw.txt id.txt normal.txt dis.txt
#python ~/myCoding/caffe/deploy/lightencnn.py --model_prototxt ../../face_verification_experiment/proto/LightenedCNN_B_deploy.prototxt --model_file  ../../face_verification_experiment/model/LightenedCNN_B.caffemodel --dump_file ./test_lfw.txt /home/robert/myCoding/suresecure/dataset/20160913-ren-zheng-bi-dui-haoyun/id_frontal_data/aligned_lfw_test 
#python ~/myCoding/caffe/deploy/lightencnn.py --model_prototxt ../../face_verification_experiment/proto/LightenedCNN_B_deploy.prototxt --model_file  ../../face_verification_experiment/model/LightenedCNN_B.caffemodel --dump_file ./id.txt ~/myCoding/suresecure/dataset/20160913-ren-zheng-bi-dui-haoyun/id_frontal_data/align_mxnet/id
#python ~/myCoding/caffe/deploy/lightencnn.py --model_prototxt ../../face_verification_experiment/proto/LightenedCNN_B_deploy.prototxt --model_file  ../../face_verification_experiment/model/LightenedCNN_B.caffemodel --dump_file ./normal.txt ~/myCoding/suresecure/dataset/20160913-ren-zheng-bi-dui-haoyun/id_frontal_data/align_mxnet/frontal_xcq
#python idverif.py --idimage-align /home/robert/myCoding/suresecure/dataset/20160913-ren-zheng-bi-dui-haoyun/id_frontal_data/align_mxnet/id --normalimage-align /home/robert/myCoding/suresecure/dataset/20160913-ren-zheng-bi-dui-haoyun/id_frontal_data/align_mxnet/frontal_xcq --model-prefix ../model/lightened_cnn/lightened_cnn --epoch 166

