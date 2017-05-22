export convnet_root=/Users/blaize/Desktop/UCSD/convNetInference
protoc -I=$convnet_root/src/proto --cpp_out=$convnet_root/src --python_out=$convnet_root/scripts $convnet_root/src/proto/network.proto
cd $convnet_root/scripts
wget http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy
python write_alexnet_pb.py
rm network_pb2.py network_pb2.pyc bvlc_alexnet.npy
cd $convnet_root
