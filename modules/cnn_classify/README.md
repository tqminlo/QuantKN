## Example train, quantize and evaluate in classify task

### Train float model
    python train.py -a TYPE_MODEL -p PRETRAINED -b BATCHSIZE -e EPOCHS -s SAVE_PATH 
###### Example:
    python train.py -a mobilenet -e 100 -s saved/mobilenet-w-1st.h5
### QAT - quantization aware training
    ###
### PTQ - quantize model
    python ptq.py -a TYPE_MODEL -w WEIGHTs_FP32 -j SAVE_JSON_PATH
###### Example:
    python ptq.py -a mobilenet -w saved/mobilenet-w-1st.h5 -j saved/mobilenet-1st.json
### Only-integer infer and evaluate
    python infer_int.py -t INFER_OR_EVAL -i INP_PATH -a TYPE_MODEL -j JSON_PATH
###### Example:
    python infer_int.py -t infer -i test/img1.jpg -a mobilenet -j saved/mobilenet-1st.json  ### infer an image
    python infer_int.py -t eval -i datasets/imagenet100/val -a mobilenet -j saved/mobilenet-1st.json  ### eval int-model






