# Inference Graph

After training the model we need to save it for deployment,saving the model means saving all the values of the parameters and the graph.

After saving the model there will be some usefull files such as:

1- tensorflowModel.ckpt.meta: Tenosrflow stores the graph structure separately from the variable values. The file .ckpt.meta contains the complete graph. It includes GraphDef, SaverDef, and so on.

2- tensorflowModel.ckpt.data-00000-of-00001: This contains the values of variables(weights, biases, placeholders, gradients, hyper-parameters etc).

3- tensorflowModel.ckpt.index: It is a table where Each key is the name of a tensor and its value is a serialized BundleEntryProto.
serialized BundleEntryProto holds metadata of the tensors. Metadata of a tensor may be like: which of the “data” files contains the content of a tensor, the offset into that file, checksum, some auxiliary data, etc.

4- checkpoint:All checkpoint information, like model ckpt file name and path
