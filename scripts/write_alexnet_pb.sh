protoc -I=$dirname/../src/proto --cpp_out=$dirname/../src/ --python_out=$dirname $dirname/../src/proto/network.proto
wget http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy
python write_alexnet_pb.py
rm network_pb2.py
