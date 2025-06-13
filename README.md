# PIMACC

PIMACC is an end-to-end accuracy simulator of computing-in-memory (CIM) architectures, which is devoted to evaluate the output accuracy of a neural network inference running on a CIM accelerator under the impact of non-ideal factors, such as quantization error, error of device resistance, IR-drop effects, etc. Error injections are flexible and can be configured in a hardware configuration file.

PIMACC works with task/data mappings and instructions given by a CIM compiler, PIMCOMP-NN, whose link is given hear: https://github.com/sunxt99/PIMCOMP-NN. The code of PIMCOMP-NN is included in this package. To run PIMACC, you need to first compile your target network model with PIMCOMP-NN, including front end and back end. Please refer to the user guide of PIMCOMP-NN to see how to use PIMCOMP-NN. To make sure you can run PIMACC after compilation, add `–v=YES` in backend compilation of PIMCOMP-NN.

# Usage

## Requirements

In order to run the PIMACC verification program, you need to install the following python package.

- onnx
- onnxruntime
- numpy
- torch
- torchvision
- cv2

## Simulation
1. Follow the usage of PIMCOMP-NN to compile the network model using frontend and backend of PIMCOMP-NN. Please note that the source code of PIMCOMP-NN is included in this package so you don't need to download PIMCOMP-NN separately. Remember to add `–v=YES` in backend compilation of PIMCOMP-NN. After compilation, PIMCOMP-NN will generate a file of `PIMACC/output/VerificationInfo.json` storing the instructions.

2. Run following commands (make sure you have the same onnx model file that is used in compilation):

```shell
cd PIMACC/verification/
python verification.py --model_path ../models/ONNX/resnet18.onnx --pipeline_type element --image_num 1000 --batchsize 10
```
PIMACC will run and give the accuracy result. This example uses the cifar10 dataset. If you want to use a different dataset, you need to provide the dataset file and slightly modify `verification.py` at line ~1070. 

## Configaration
Non-ideal effects are configured in **PIMACC/config.json**, you can set values of the following non-ideal effects.
- "cell_precision": cell precision, how many bits a cell can store
- "reference_conductance_state": cell conductance (siemens) of low-resistance-state
- "R_ratio": on-off ratio, the ratio of high and low resistances
- "bitline_conductance": wire conductance (siemens) between two adjacent devices on bitline
- "wordline_conductance": wire conductance (siemens) between two adjacent devices on wordline
- "variation": device resistance variation, which is the relative deviation and calculated as (standard deviation)/(ideal value)
- "SAF_flag": whether to consider stuck-at faults
- "p_SA0": probability of stuck-at-0 (high-resistance state)
- "p_SA1": probability of stuck-at-1 (low-resistance state)

# Code Author

Haocheng Han (Institute of Computing Technology, Chinese Academy of Sciences)

# Project PI

[Xiaoming Chen](https://people.ucas.edu.cn/~chenxm)